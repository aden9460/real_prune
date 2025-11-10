import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed

from slim_utils.slimgpt import SlimGPT
from slim_utils.slim_dataset import get_loaders
from slim_utils.params_remove import LLaMAParamsPruner
from ppl_eval.ppl_eval import ppl_metric


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_model(model_dir):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import LlamaForCausalLM, LlamaTokenizer
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
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


@torch.no_grad()
def model_slimming(model, dataloader, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype

    print("preparing...")
    model.model.embed_tokens = model.model.embed_tokens.cuda()
    model.model.norm = model.model.norm.cuda()

    layers = model.model.layers
    layer0 = layers[0]
    layers[0] = Catcher(
        model.seqlen, 
        model.config.hidden_size, 
        args.num_samples, 
        args.batch_samples, 
        args.cache_dev, 
        dtype
    )

    inputs_len = []
    for data in dataloader:
        # inp, inp_len = data
        # inputs_len.append(inp_len)
        inp = data
        inputs_len.append(model.seqlen)
        try:
            model(inp.cuda())
        except ValueError:
            pass
    layer_inputs = layers[0].layer_inputs
    attention_mask = layers[0].attention_mask
    layers[0] = layer0
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    print("pruning...")
    for i in range(len(layers)):
        layer = layers[i].cuda()
        
        if args.minlayer <= i < args.maxlayer:
            all_module_dict = find_layers(layer)
            # print(all_module_dict.keys())

            sequential = [
                ["self_attn.o_proj"],
                ["mlp.down_proj"],
            ]
            for names in sequential:
                module_dict = {name: all_module_dict[name] for name in names}
                pruner_dict = {}
                for name in module_dict:
                    pruner_dict[name] = SlimGPT(module_dict[name], i, args)  # init

                def add_batch(name):
                    def func(_, inp, out):
                        pruner_dict[name].add_batch(inp[0].data, out.data)  # calculate H
                    return func

                handles = []
                for name in module_dict:
                    # The hook will be called every time after forward() has computed an output
                    handles.append(module_dict[name].register_forward_hook(add_batch(name)))
                if args.cache_dev == 'cuda':
                    for j in range(args.num_samples):
                        layer(layer_inputs[j][: inputs_len[j]].unsqueeze(0), attention_mask=attention_mask)  # forward
                else:
                    for j in range(0, args.num_samples, args.batch_samples):
                        inps_num = min(args.num_samples - j, args.batch_samples)
                        inps = layer_inputs[j: j + inps_num].cuda()
                        inps_len = inputs_len[j: j + inps_num]
                        for k in range(inp_num):
                            layer(inps[k][: inps_len[k]].unsqueeze(0), attention_mask=attention_mask)
                for h in handles:
                    h.remove()

                for name in module_dict:
                    print(f"layer {i}: {name}")
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    pruner_dict[name].struct_prune(
                        sparsity=sparsity,
                        percdamp=args.percdamp,
                        headsize=model.config.hidden_size // model.config.num_attention_heads if name == "self_attn.o_proj" else 1,
                        layer_idx=i,
                    )
                    pruner_dict[name].free()

            del pruner_dict
        
        if args.cache_dev == 'cuda':
            for j in range(args.num_samples):
                layer_inputs[j] = layer(layer_inputs[j: j+1], attention_mask=attention_mask)[0]  # inplace op
        else:
            for j in range(0, args.num_samples, args.batch_samples):
                inp_num = min(args.num_samples - j, args.batch_samples)
                inp = layer_inputs[j: j + inp_num].cuda()
                for k in range(inp_num):
                    inp[k] = layer(inp[k: k+1], attention_mask=attention_mask)[0]
                layer_inputs[j: j + inp_num] = inp.cpu()

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache


def main(args):
    print('load model...')
    model, tokenizer = get_model(args.model_path)
    model.seqlen = args.seqlen
    model.eval()
    
    args.minlayer = max(args.minlayer, 0)
    args.maxlayer = min(args.maxlayer, model.config.num_hidden_layers)

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
    dataloader = get_loaders(
        args.dataset, 
        num_samples=args.num_samples, 
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )

    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')
    
    if isinstance(args.sparsity, list) or args.sparsity > 0 :
        print('start slimming...')
        tick = time.time()
        model_slimming(model, dataloader, args)
        print(time.time() - tick)

    if args.save_dir:
        print('saving model...')
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
    
        if args.save_pruned_weights:
            print('saving pruned model...')
            params_pruner = LLaMAParamsPruner(model)
            pruned_state_dict, layer_head_num, layer_inter_size = params_pruner.remove_params()
            model.config.layer_head_num = layer_head_num
            model.config.layer_inter_size = layer_inter_size
            pruned_save_dir = os.path.join(args.save_dir, 'pruned_weights')
            model.save_pretrained(pruned_save_dir, state_dict=pruned_state_dict)
            tokenizer.save_pretrained(pruned_save_dir)

    if not args.skip_evaluate:
        print('start evaluate...')
        ppl_metric(model.cuda().half(), tokenizer, ['wikitext2'], 128, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path", type=str, help="model to load"
    )
    parser.add_argument(
        "dataset", type=str,
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
        "--sparsity", type=float, default=0, 
        help="Target pruning ratio, which does not take effect when non_uniform is True"
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, 
        help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, 
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
        "--min_sparsity", type=float, default=0,
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--max_sparsity", type=float, default=0,
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

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)