from pathlib import Path
import sys
lib_path = str(Path(__file__).absolute().parent)
print(lib_path)
sys.path.append(lib_path)

import os
import torch


# 'model.layers.31.self_attn.q_proj.weight'
# 'model.layers.31.self_attn.k_proj.weight'
# 'model.layers.31.self_attn.v_proj.weight'
# 'model.layers.31.self_attn.o_proj.weight'
# 'model.layers.31.self_attn.rotary_emb.inv_freq'
# 'model.layers.31.mlp.gate_proj.weight'
# 'model.layers.31.mlp.up_proj.weight'
# 'model.layers.31.mlp.down_proj.weight'
# 'model.layers.31.input_layernorm.weight'
# 'model.layers.31.post_attention_layernorm.weight'


class LLaMAParamsPruner(object):
    def __init__(self, model) -> None:
        self.model = model
        self.suffix_size = 3
        self.column_pruning_suffix = [
            'self_attn.o_proj.weight',
            'mlp.down_proj.weight'
        ]
        self.row2column_proj = {
            'self_attn.q_proj.weight': 'self_attn.o_proj.weight',
            'self_attn.k_proj.weight': 'self_attn.o_proj.weight',
            'self_attn.v_proj.weight': 'self_attn.o_proj.weight',
            'mlp.gate_proj.weight': 'mlp.down_proj.weight',
            'mlp.up_proj.weight': 'mlp.down_proj.weight',
        }
        self.row_pruning_suffix = set(self.row2column_proj.keys())
        self.head_num_ref_name = 'self_attn.o_proj.weight'
        self.ffn_dim_ref_name = 'mlp.down_proj.weight'
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

        self._init_sparse_config()

    def _init_sparse_config(self):
        sparse_config = {}
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            name_suffix = '.'.join(name.split('.')[-self.suffix_size:])
            if name_suffix in self.column_pruning_suffix:
                sparse_config[name] = param.abs().sum(0) > 0
        # print(sparse_config.keys())
        self.sparse_config = sparse_config

    def _get_sparse_masks(self, name):
        sparse_masks = []
        suffix = '.'.join(name.split('.')[-self.suffix_size:])
        if suffix in self.column_pruning_suffix:
            sparse_mask = self.sparse_config[name]
            sparse_masks.append(sparse_mask.unsqueeze(0))
        
        if suffix in self.row2column_proj:
            row2column_suffix = self.row2column_proj[suffix]
            sparse_mask = self.sparse_config[name.replace(suffix, row2column_suffix)]
            # TODO be careful about the reshape rules!!!
            sparse_masks.append(sparse_mask.unsqueeze(1))  
        return sparse_masks
    
    def remove_params(self):
        state_dict = self.model.state_dict()
        pruned_state_dict = {}
        layer_head_num, layer_inter_size = [], []
        for name, param in state_dict.items():
            sparse_masks = self._get_sparse_masks(name)
            pruned_param = param.clone()
            for sparse_mask in sparse_masks:
                if sparse_mask.size(0) == 1:  # columns
                    pruned_param = pruned_param.masked_select(sparse_mask).reshape(pruned_param.size(0), -1)
                elif sparse_mask.size(1) == 1:  # rows
                    if pruned_param.ndim == 1:  # bias
                        pruned_param = pruned_param.masked_select(sparse_mask.squeeze())
                    else:
                        pruned_param = pruned_param.masked_select(sparse_mask).reshape(-1, pruned_param.size(1))
            pruned_state_dict[name] = pruned_param
            print(f'{name}\toriginal size: {param.size()}\tpruned size: {pruned_param.size()}', flush=True)

            suffix = '.'.join(name.split('.')[-self.suffix_size: ])
            if suffix == self.head_num_ref_name:
                layer_head_num.append(pruned_param.size(-1) // self.head_dim)
            elif suffix == self.ffn_dim_ref_name:
                layer_inter_size.append(pruned_param.size(-1))
        
        print('params: ', round(sum(p.numel() for p in state_dict.values()) / 10**9, 2), 'B')
        print('after pruned params: ', round(sum(p.numel() for p in pruned_state_dict.values()) / 10**9, 2), 'B')
        return pruned_state_dict, layer_head_num, layer_inter_size
    
    def mask_params(self):
        state_dict = self.model.state_dict()
        masked_state_dict = {}
        for name, param in state_dict.items():
            sparse_masks = self._get_sparse_masks(name)
            masked_param = param.clone()
            for sparse_mask in sparse_masks:
                if masked_param.ndim == 1 and sparse_mask.size(1) == 1:  # bias & rows
                    masked_param *= sparse_mask.squeeze()
                else:
                    masked_param *= sparse_mask
            masked_state_dict[name] = masked_param
            
        return masked_state_dict


class QwenParamsPruner(LLaMAParamsPruner):
    def __init__(self, model) -> None:
        self.model = model
        self.suffix_size = 3
        self.column_pruning_suffix = [
            'attn.c_proj.weight',
            'mlp.c_proj.weight'
        ]
        self.row2column_proj = {
            'attn.c_attn.weight': 'attn.c_proj.weight',
            'attn.c_attn.bias': 'attn.c_proj.weight',
            'mlp.w1.weight': 'mlp.c_proj.weight',
            'mlp.w2.weight': 'mlp.c_proj.weight',
        }
        self.row_pruning_suffix = set(self.row2column_proj.keys())
        self.head_num_ref_name = 'attn.c_proj.weight'
        self.ffn_dim_ref_name = 'mlp.c_proj.weight'
        self.head_dim = model.config.kv_channels

        self._init_sparse_config()

    def _get_sparse_masks(self, name):
        sparse_masks = []
        suffix = '.'.join(name.split('.')[-self.suffix_size: ])
        if suffix in self.column_pruning_suffix:
            sparse_mask = self.sparse_config[name]
            sparse_masks.append(sparse_mask.unsqueeze(0))
        
        if suffix in self.row2column_proj:
            row2column_suffix = self.row2column_proj[suffix]
            sparse_mask = self.sparse_config[name.replace(suffix, row2column_suffix)]
            if row2column_suffix == self.head_num_ref_name:
                sparse_mask = sparse_mask.unsqueeze(0).expand(3, -1).reshape(-1)
                sparse_masks.append(sparse_mask.unsqueeze(1))
            elif row2column_suffix == self.ffn_dim_ref_name:
                sparse_masks.append(sparse_mask.unsqueeze(1))
                
        return sparse_masks
