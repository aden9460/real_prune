from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import os
from torch import nn
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.training_args import OptimizerNames
from transformers.optimization import AdamW as AdamWHF
from abc import ABCMeta, abstractmethod
import json


logger = logging.get_logger(__name__)

# model.layers.31.self_attn.q_proj.weight
# model.layers.31.self_attn.k_proj.weight
# model.layers.31.self_attn.v_proj.weight
# model.layers.31.self_attn.o_proj.weight
# model.layers.31.self_attn.sparse_mask.weight
# model.layers.31.mlp.gate_proj.weight
# model.layers.31.mlp.up_proj.weight
# model.layers.31.mlp.down_proj.weight
# model.layers.31.mlp.sparse_mask.weight
# model.layers.31.input_layernorm.weight
# model.layers.31.post_attention_layernorm.weight
# model.norm.weight
# lm_head.weight


class LLaMAMaskTrainer(Seq2SeqTrainer):
    def __init__(self, model=None, args=None, sp_args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None), preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.sp_args = sp_args
        self.head_size = 128
        self.mask_config = None
        if sp_args.mask_config_dir:
            self.mask_config = json.load(open(sp_args.mask_config_dir, 'r'))
        if self.args.local_rank == 0:
            print(self.sp_args)

    def _is_columns_pruned_module(self, name):
        return name.endswith('self_attn.o_proj') or name.endswith('mlp.down_proj')
    
    def _is_attention_module(self, name):
        return name.endswith('self_attn.o_proj')

    def _mask_sparse_weight(self, model: nn.Module):
        sparse_mask_dict = {}
        for name, m in model.named_modules():
            if self._is_columns_pruned_module(name):

                if getattr(m, 'sparse_mask', None) is None:
                    if self.mask_config:
                        mask = torch.ones(m.weight.size(1), dtype=m.weight.data.dtype, device=m.weight.data.device)
                        key = name.replace('base_model.model.', '') + '.weight'
                        value = self.mask_config[key]
                        if self._is_attention_module(name) and len(value) > 0:
                            value = torch.hstack([torch.arange(i * self.head_size, (i+1) * self.head_size) for i in value])
                        mask[value] = 0
                    else:
                        mask = (m.weight.data.abs().sum(0) > 0.1).to(m.weight.data.dtype)
                    m.register_buffer('sparse_mask', mask)
                    sparse_mask_dict[name] = mask
                    if self.args.local_rank == 0:
                        print('name:%s\tweight_num:%s\tpruned_num:%s\tpruned_ratio:%s' % (
                            name, mask.size(-1), (1-mask).sum().item(), round((1-mask).sum().item() / mask.size(-1), 4)
                        ))
                        if self._is_attention_module(name):
                            print(torch.unique(mask.float().view(-1, 128).sum(1)))
                    
                m.weight.data.mul_(m.sparse_mask)
                if getattr(m, 'bias', None) is not None:
                    m.bias.data.mul_(m.sparse_mask)
        
        # lora mask
        for name, m in model.named_modules():
            if name.endswith('lora_A.default'):
                base_name = '.'.join(name.split('.')[:-2])
                if base_name in sparse_mask_dict:

                    if getattr(m, 'sparse_mask', None) is None:
                        mask = sparse_mask_dict[base_name]
                        m.register_buffer('sparse_mask', mask)

                    m.weight.data.mul_(m.sparse_mask)
                    if getattr(m, 'bias', None) is not None:
                        m.bias.data.mul_(m.sparse_mask)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # forward pass
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            raise Exception('apex is NOT supported!')
        else:
            self.accelerator.backward(loss)  # backward and engine both step here -.-!
        
        # Gradients are only available immediately after backward and before engine step
        if self.sp_args.is_mask_training:
            self._mask_sparse_weight(unwrap_model(model))

        return loss.detach() / self.args.gradient_accumulation_steps
