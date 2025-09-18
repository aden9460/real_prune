 ###由于全尺度评估需要的输入较多 会爆显存 所以进行更改 逐block即时获得hessian 算完释放
#发现代码不需要这么复杂 slimgpt的钩子写好之后 直接前向传播即可
import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp

from slim_utils.slimgpt import SlimGPT
from slim_utils.slim_dataset import get_loaders
from slim_utils.params_remove import LLaMAParamsPruner

from torchvision.utils import save_image
import sys
sys.path.append("/home/suanba/EdgeVAR/Torch-Pruning")
from importlib.metadata import version
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import os
import torch_pruning as tp
from torchviz import make_dot
import torch_pruning.pruner.function as tfun
from models import VQVAE, build_vae_var
import gc
from contextlib import contextmanager
from torch.utils.data import DataLoader, TensorDataset


# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False

    # layers = model.model.layers
    layers = model.blocks
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    # model.config.use_cache = use_cache
    return float(count) / total_params




def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


# @torch.no_grad()
def model_slimming(model, dataloader, args):

    batch = len(dataloader)

    

    dev = "cuda" if torch.cuda.is_available() else 'cpu'

    layers = model.blocks


    with torch.no_grad():

        t1 = time.time()
        # inps, outs, cond_BD, cache = prepare_calibration_d16_last_input(
        #     args, model, dataloader, dev )
        num_batches = len(dataloader)
        print("pruning...")
        caches = [None for _ in range(num_batches*10)]
        outs = [None for _ in range(num_batches*10)] 
        for i in range(len(layers)):
            
            layer = layers[i].to(dev)
            if args.minlayer <= i < args.maxlayer:
                all_module_dict = find_layers(layer)

            sequential = [####################################
                ["attn.proj"],
                ["ffn.fc2"],
            ]
            for names in sequential:
                module_dict = {name: all_module_dict[name] for name in names}
                pruner_dict = {}
                for name in module_dict:
                    pruner_dict[name] = SlimGPT(module_dict[name], i, args)  # 对每一个层都初始化一个剪枝器

                def add_batch(name):
                    def func(_, inp, out):
                        pruner_dict[name].add_batch(inp[0].data, out.data)  # calculate H
                    return func




                handles = []
                for name in module_dict:
                    handles.append(module_dict[name].register_forward_hook(add_batch(name))) #注册钩子
                for b in model.blocks: b.attn.kv_caching(True)

                for batch_idx, batch in enumerate(dataloader):
                    model(batch)
                for b in model.blocks: b.attn.kv_caching(False)
                for h in handles:
                    h.remove()
                
                for name in module_dict:
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    print(f"layer {i}: {name} sparsity {sparsity}")
                    if args.prune_method == "slimgpt":
                        idx = pruner_dict[name].struct_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "magnitude":
                        idx = pruner_dict[name].magnitude_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "taylor":
                        idx = pruner_dict[name].taylor_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    pruner_dict[name].free()
                    target_layer = get_module_by_name(model.blocks[i], name)
                    if name == "ffn.fc2":
                        target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                        idx = idx.tolist()
                        tp.prune_linear_in_channels(target_layer,idx)
                        tp.prune_linear_out_channels(target_layer_b,idx)
                    elif name == "attn.proj" :
                        # model.blocks[i].attn.pruned_indices.copy_(idx.int())
                        sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                        model.blocks[i].attn.num_heads = torch.round(torch.tensor(model.num_heads*(1-sparsity))).int()
                        idx_m = idx.to(dtype=torch.long)    
                        idx = idx.tolist()
                        keep_idxs = list(set(range(target_layer.in_features)) - set(idx))
                        model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
                        zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
                        model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
                        model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])
                        model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
                        torch.full(size=(1, model.blocks[i].attn.num_heads, 1, 1),fill_value=4.0,device='cuda').log(),requires_grad=True)
                        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
                        tp.prune_linear_in_channels(target_layer, idx) #proj的inchannel
                        
                        hidden = 16 * 64

                        rm_feat_q = idx_m                                # 直接就是单段里的通道
                        # 映射到合并 QKV 的 out 索引（mat_qkv 的第0维）
                        rm_qkv = torch.cat([rm_feat_q,
                                            rm_feat_q + hidden,
                                            rm_feat_q + 2*hidden], dim=0)   # [3m]

                        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
                        tp.prune_linear_out_channels(target_layer_b,rm_qkv_list )

            del pruner_dict   
            print(model.blocks[i].ffn.fc1.weight.shape)
            print(model.blocks[i].ffn.fc2.weight.shape)
            print(model.blocks[i].attn.mat_qkv.weight.shape)
            print(model.blocks[i].attn.proj.weight.shape)

            del layer
            torch.cuda.empty_cache()

    print("本次评估用用时为{}".format(time.time()-t1))
    
    return model




def main(args):
    print('load model...')
    # model, tokenizer = get_model(args.model_path)
    MODEL_DEPTH =  args.maxlayer   # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {12,16, 20, 24, 30}
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = '/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth', f'/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): print("var not exist")
    if not osp.exists(var_ckpt): print("var not exist")
    # if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    # if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False
        )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')
    model = var
    tokenizer = None

    # # model.seqlen = args.seqlen
    model.eval()
    
    args.minlayer = max(args.minlayer, 0)
    args.maxlayer = min(args.maxlayer,36)

    if args.non_uniform:
        assert 0 <= args.min_sparsity <= args.max_sparsity < 1        
        if args.non_uniform_strategy in ('log_increase', 'log_decrease'):
            linear_space = np.arange(0, args.maxlayer - args.minlayer)
            args.sparsity = args.min_sparsity + (args.max_sparsity - args.min_sparsity) / np.log(32) * np.log(1 + linear_space)
            args.sparsity = [0] * args.minlayer + list(args.sparsity)
            if args.non_uniform_strategy == 'log_decrease':
                args.sparsity = args.sparsity[::-1]
        elif args.non_uniform_strategy in ('linear_increase', 'linear_decrease'):
            sparsity_grad = (args.max_sparsity - args.min_sparsity) / (args.maxlayer - 1 - args.minlayer)
            args.sparsity = [(i - args.minlayer) * sparsity_grad + args.min_sparsity for i in range(args.minlayer, args.maxlayer)]
            args.sparsity = [0] * args.minlayer + args.sparsity
            if args.non_uniform_strategy == 'linear_decrease':
                args.sparsity = args.sparsity[::-1]
        else:
            raise Exception

    state_dict = model.state_dict()
    # print(state_dict.keys())
    layer_params = round(sum(v.numel() for k,v in state_dict.items() if k not in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k,v in state_dict.items() if k in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    # sparsity_avg = sum(args.sparsity) / len(args.sparsity) if isinstance(args.sparsity, list) else args.sparsity
    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')

    print('load dataset...')

    # dataloader = torch.arange(0,args.num_samples).cuda()
    dataset = torch.arange(0, args.num_samples)
    # 包装成 TensorDataset
    tensor_dataset = TensorDataset(dataset)
    # 创建 DataLoader，每次 batch_size=10，按顺序不打乱
    dataloader = DataLoader(tensor_dataset, batch_size=10, shuffle=False)
    # dataloader = torch.randint(0, 1000, (args.num_samples,)).cuda()

    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')
    
    if isinstance(args.sparsity, list) or args.sparsity >= 0 :

        print('start slimming...')
        tick = time.time()
        model= model_slimming(model, dataloader, args)
        print(time.time() - tick)
        
    
    print("*"*30)
    sparsity_ratio = check_sparsity(model)

    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    # print(model)
    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k,v in state_dict.items() if k not in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k,v in state_dict.items() if k in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    # sparsity_avg = sum(args.sparsity) / len(args.sparsity) if isinstance(args.sparsity, list) else args.sparsity
    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')
    example_input = torch.tensor([0]).to(device)
    for b in model.blocks: b.attn.kv_caching(True)
    torch.cuda.synchronize()
    zero = time.time()
    result = var(example_input)
    torch.cuda.synchronize()
    end = time.time()
    print("all:{}",end-zero)

    print(model)
    macs, nparams = tp.utils.count_ops_and_params(model, example_input, layer_wise=False)

    print(model(example_input).shape)
    print(
        "  Params: %.2f M => %.2f M"
        % ( nparams / 1e6, nparams / 1e6)
    )
    print(
        "   MACs: %.2f G => %.2f G"
        % ( macs / 1e9, macs / 1e9)
    )


    save_model = "/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/sparsity_model"
    os.makedirs(save_model,exist_ok=True)
    save_path = os.path.join(save_model, args.model_name)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, 
        default="/home/sumingluo/Model_weight/meta/llama-2-7b-hf", help="model to load"
    )
    parser.add_argument(
        "--dataset", type=str,default="wikitext2",
        choices=["wikitext2", "c4", "alpaca", "gpt4_alpaca"],
        help="Where to extract calibration data from.",
    )
    
    parser.add_argument(
        "--num_samples", type=int, default=1024, 
        help="Number of calibration data samples."
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048, 
        help="Sequence length for the calibration data."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.2, 
        help="Target pruning ratio, which does not take effect when non_uniform is True"
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, 
        help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=32, 
        help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--cache_dev", type=str, default="cuda", 
        help="Defaults to `cuda`. When the GPU memory is insufficient, you can set `cache_dev` to `cpu`, but the trade-off is slower pruning speed."
    )
    parser.add_argument(
        "--batch_samples", type=int, default=128, 
        help="Works when `cache_dev=cpu`. The number of samples loaded onto the GPU each time."
    )
    parser.add_argument(
        "--skip_evaluate", action="store_true",
        help="When set to True, skip the evaluation on Wikitext-2 after the pruning is complete.",
    )
    parser.add_argument(
        "--save_pruned_weights", action="store_true",
        help="Whether save the checkpoint after removing the zeroed-out parameters.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="", 
        help="Path to saved model.",
    )

    # slimgpt & non_uniform config
    parser.add_argument(
        "--non_uniform", action="store_true",
        help="When set to True, use non-uniform pruning, and the parameter sparsity will be ineffective.",
    )
    parser.add_argument(
        "--non_uniform_strategy", type=str, default='log_increase', 
        choices=["log_increase", "log_decrease", "linear_increase", "linear_decrease"],
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--min_sparsity", type=float, default=0.06,
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--max_sparsity", type=float, default=0.3,
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--no_compensate", action="store_true",
        help="Skip error compensation in SlimGPT",
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, 
        help="Seed for sampling the calibration data."
    ) 
    parser.add_argument(
        "--prune_method", type=str, default="slmipgpt", 
        help="method.",
    )
    parser.add_argument(
        "--specific_layer", type=int, default=0, 
        help="choose which layer to evaluate",
    )

    parser.add_argument(
        "--model_name", type=str, default="model", 
        help="choose which layer to evaluate",
    )
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)
