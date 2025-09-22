import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
from eval import eval_ppl
from DFOBS.slim_utils.slimgpt import SlimGPT
from slim_utils.slim_dataset import get_loaders
from slim_utils.params_remove import LLaMAParamsPruner
from ppl_eval.ppl_eval import ppl_metric
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


def get_model(model_dir):
    # def skip(*args, **kwargs):
    #     pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    model =  LlamaForCausalLM.from_pretrained(
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        pretrained_model_name_or_path = model_dir
    )
    model.seqlen = 2048

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_dir, 
                                              use_fast=False)
    tokenizer.bos_token = '<s>'  # token_id 1
    tokenizer.eos_token = tokenizer.pad_token = tokenizer.unk_token = '</s>'  # token_id 2 
    return model, tokenizer


class Catcher(nn.Module):
    def __init__(self, seqlen, hidden_size, num_samples, batch_samples, cache_dev='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.layer_inputs = torch.zeros(
            (num_samples, seqlen, hidden_size), 
            dtype=dtype, device=cache_dev
        )
        if cache_dev == 'cpu':
            self.batch_inputs = torch.zeros(
                (batch_samples, seqlen, hidden_size), 
                dtype=dtype, device='cuda'
            )
        self.batch_samples = batch_samples
        self.row_idx = 0
        self.batch_idx = 0
        self.attention_mask = None
        self.cache_dev = cache_dev

    def forward(self, inputs, **kwargs):
        if self.cache_dev == 'cpu':
            self.batch_inputs[self.row_idx] = inputs
            self.row_idx += 1
            if self.row_idx == self.batch_samples:
                self.layer_inputs[self.batch_idx: self.batch_idx + self.batch_samples] = self.batch_inputs.to(self.cache_dev)
                self.row_idx = 0
                self.batch_idx += 1
        else:
            self.layer_inputs[self.row_idx] = inputs
            self.row_idx += 1            

        if self.attention_mask is None and kwargs["attention_mask"] is not None:
            self.attention_mask = kwargs["attention_mask"]

        raise ValueError


def prepare_calibration_input(args, model, dataloader, device="cuda"):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.num_samples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prepare_calibration_d16_last_input(args, model, dataloader,device):
    
    label_B = dataloader.int()

    layers = model.blocks

    # inps = torch.zeros(
    #     (len(label_B), 256, 1024),device="cuda")
    
    # for b in layers: b.attn.kv_caching(False)
    # inps.requires_grad = False
    # cache = {"i": 0,"x":[], "cond_BD": [], "attn_bias": []}
    # outs = {"i": 0,"x":[], "cond_BD": [], "attn_bias": []}
    cache = []
    outs = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, **kwargs):  ##ori 只选取最后一层
            # inps[cache["i"]] = inp  #
            
            # cache["cond_BD"] = kwargs["cond_BD"]
            # cache["attn_bias"] = kwargs["attn_bias"]
            # inps.append(inp)
            # print(x.shape)
            tokens = x.shape[1]
            if tokens ==args.specific_layer :
                cache.append({
                "i": len(cache),
                "x": x,
                "cond_BD": kwargs["cond_BD"],

                })
                outs.append({
                "i": len(cache),
                "x": 0,
                "cond_BD": kwargs["cond_BD"],

                })
                raise ValueError
            else:
                return self.module(x, **kwargs)

        # def forward(self, x, **kwargs): #每一层都要 _allstep
        #     # inps[cache["i"]] = inp  #
            
        #     # cache["cond_BD"] = kwargs["cond_BD"]
        #     # cache["attn_bias"] = kwargs["attn_bias"]
        #     # inps.append(inp)
        #     # print(x.shape)
        #     # tokens = x.shape[1]
        #     # if tokens ==256 :
        #     cache.append({
        #     "i": len(cache),
        #     "x": x,
        #     "cond_BD": kwargs["cond_BD"],

        #     })
        #     outs.append({
        #     "i": len(cache),
        #     "x": 0,
        #     "cond_BD": kwargs["cond_BD"],

        #     })
        #     raise ValueError
        #     # else:
        #         # return self.module(x, **kwargs)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    # outs = torch.zeros_like(inps)

    # position_ids = None
    cond_BD_or_gss = None
    return cond_BD_or_gss, outs, cond_BD_or_gss, cache

def prepare_calibration_d16_last_input(args, model, dataloader,device):
    
    label_B = dataloader.int()

    layers = model.blocks

    # inps = torch.zeros(
    #     (len(label_B), 256, 1024),device="cuda")
    
    # for b in layers: b.attn.kv_caching(False)
    # inps.requires_grad = False
    # cache = {"i": 0,"x":[], "cond_BD": [], "attn_bias": []}
    # outs = {"i": 0,"x":[], "cond_BD": [], "attn_bias": []}
    cache = []
    outs = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, **kwargs):  ##ori 只选取最后一层
            # print(x.shape)
            # print("layer_id:{}".format(self.layer_id))
            tokens = x.shape[1]
            # if tokens ==args.specific_layer[layer_id]**2:
            if tokens ==args.specific_layer:
                cache.append({
                "i": len(cache),
                "x": x,
                "cond_BD": kwargs["cond_BD"],

                })
                outs.append({
                "i": len(cache),
                "x": 0,
                "cond_BD": kwargs["cond_BD"],

                })
                raise ValueError
            else:
                return self.module(x, **kwargs)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module


    cond_BD_or_gss = None
    return cond_BD_or_gss, outs, cond_BD_or_gss, cache

def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


# @torch.no_grad()
def model_slimming(model, dataloader, args):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    # dtype = next(iter(model.parameters())).dtype

    # print("preparing...")
    # model.model.embed_tokens = model.model.embed_tokens.cuda()
    # model.model.norm = model.model.norm.cuda()

    # layers = model.model.layers
    batch = len(dataloader)

    

    dev = "cuda" if torch.cuda.is_available() else 'cpu'

    # out = model(torch.LongTensor([0,1,2,3]).cuda())

    # dot = make_dot(out, params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render('computational_graph')  # 会生成 computational_graph.png
    # dot = make_dot(out.sum(), params=dict(model.named_parameters()))
    # dot.save('computational_graph.dot')
    # DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.LongTensor([0,1,2,3]).cuda())
    # for group in DG.get_all_groups(root_module_types=[nn.Conv2d, nn.Linear]):
    # # Handle groups in sequential order
    #     idxs = [2,4,6] # your pruning indices, feel free to change them
    #     group.prune(idxs=idxs)
    #     print(group)
    layers = model.blocks

    # if "model.embed_tokens" in model.hf_device_map:
    #     dev = model.hf_device_map["model.embed_tokens"]
# with torch.no_grad():
    # inps, outs, attention_mask, position_ids = prepare_calibration_input(
    #     args, model, dataloader, dev
    # )
    # out = model(torch.LongTensor([0,1,2,3]).cuda())
    # dot = make_dot(out.sum(), params=dict(model.named_parameters()))
    # dot.save('computational_graph.dot')
    t1 = time.time()
    inps, outs, cond_BD, cache = prepare_calibration_d16_last_input(
        args, model, dataloader, dev )

    print("pruning...")
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
            for j in range(batch):  #前向传播 正向推理 钩子会记录输出
                outs[j]["x"] = layer(
                        cache[j]["x"],
                        cond_BD=cache[j]["cond_BD"],
                        attn_bias=None,)
                loss = outs[j]["x"].sum()
            loss.backward(retain_graph=True)
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
                model.zero_grad()
                target_layer = get_module_by_name(model.blocks[i], name)
                if name == "ffn.fc2":
                    target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                    idx = idx.tolist()
                    tp.prune_linear_in_channels(target_layer,idx)
                    tp.prune_linear_out_channels(target_layer_b,idx)
                elif name == "attn.proj" :

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
        # print(model)
        for b in model.blocks: b.attn.kv_caching(True)                    
        if args.cache_dev == 'cuda':
            for j in range(batch):  #前向传播 正向推理 钩子会记录输出
                outs[j]["x"] = layer(
                        cache[j]["x"],
                        cond_BD=cache[j]["cond_BD"],
                        attn_bias=None,
                    )
        for b in model.blocks: b.attn.kv_caching(False)   
        # layers[i] = layer
        del layer
        torch.cuda.empty_cache()
        # inps, outs = outs, inps
        # cache, outs = outs, cache
        cache = outs
        
    # model.config.use_cache = use_cache
    # for b in layers: b.attn.kv_caching(True)
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
    for p in vae.parameters(): p.requires_grad_(True)
    for p in var.parameters(): p.requires_grad_(True)
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
    # dataloader = get_loaders(
    #     args.dataset, 
    #     num_samples=args.num_samples, 
    #     seqlen=model.seqlen,
    #     tokenizer=tokenizer
    # )
    # dataloader = torch.arange(0,args.num_samples).cuda()

    dataloader = torch.randint(0, 1000, (args.num_samples,)).cuda()

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


    # save_dir = "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/d24_test_0.2_200i_temporary"
    # os.makedirs(save_dir,exist_ok=True)
    save_model = "/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/sparsity_model"
    os.makedirs(save_model,exist_ok=True)
    save_path = os.path.join(save_model, args.model_name)
    torch.save(model.state_dict(), save_path)

    # # dot_path = os.path.join(save_model, '0.4var_1i_256input_computational_graph.dot')
    # # dot = make_dot(result, params=dict(model.named_parameters()))
    # # dot.save(dot_path)
    # # for class_num in range(5):
    # class_num = 0
    # class_labels = (980, 980, 437, 437, 22, 22, 562, 562,50,50)
    # # class_labels = (980,)
    # # class_labels  = torch.full((10,), class_num, dtype=torch.long).cuda()
    # label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    # with torch.inference_mode():
    #     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
    #         for b in model.blocks: b.attn.kv_caching(True)
    #         recon_B3HW = model(label_B)
    #         for i in range(recon_B3HW.shape[0]):
    #             save_image(recon_B3HW[i], f'{save_dir}/{class_num:03d}_img_{i:03d}.png')

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)


    # with torch.inference_mode():
    #     B = len(class_labels)
    #     label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    #     with measure_peak_memory():
    #         with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
    #             for i in range(3):
    #                 start_event.record()
    #                 recon_B3HW = model(label_B)
    #                 end_event.record()
    #                 torch.cuda.synchronize()
    #                 # Calculation run time (milliseconds)
    #                 elapsed_time = start_event.elapsed_time(end_event)
    #                 print("running time:",int(elapsed_time),"ms", "batch size:",str(len(class_labels)))

    # if not args.skip_evaluate:
    #     print('start evaluate...')
    #     dataset = 'wikitext2'
    #     ppl = eval_ppl(model.cuda(), tokenizer, dataset, device=torch.device("cuda:0"))
    #     print(f"\nppl on {dataset}: {ppl}\n")
    #     ppl_metric(model.cuda().half(), tokenizer, ['wikitext2'], 128, 2)

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
#     parser.add_argument(
#     "--specific_layer", type=int, nargs='+', default=[0],
#     help="choose which layer(s) to evaluate, e.g. --specific_layer 0 1 2"
# )
    parser.add_argument(
        "--model_name", type=str, default="model", 
        help="choose which layer to evaluate",
    )
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)
