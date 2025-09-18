import os
from typing import Optional, Dict
from dataclasses import dataclass, field
from transformers.utils import logging

import torch
import transformers
from transformers.training_args import TrainingArguments
from datasets import load_dataset 

from train_utils.sparse_trainer import LLaMAMaskTrainer
from train_utils.finetune_dataset import get_loaders
from ppl_eval.ppl_eval import ppl_metric


logger = logging.get_logger(__name__)

def set_logger_level():
    """Only print log in local rank 0."""
    is_local_first_process = int(os.environ.get("LOCAL_RANK", "0")) == 0

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_local_first_process else logging.WARN,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

    #quantization
    quantization_bit: Optional[int] = field(
        default=None
    )

    #lora param
    use_lora:  Optional[bool] = field(
        default=False,
        metadata={"help": "use lora"}
    )
    peft_lora_r: int = field(
        default=8, metadata={"help": "Lora attention dimension"}
    )
    peft_lora_alpha: int = field(
        default=16, metadata={"help": "Lora alpha"}
    )
    peft_lora_dropout: float = field(
        default=0.05, metadata={"help": "Lora dropout"}
    )
    peft_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
    )
    peft_lora_bias: str = field(
        default="none"
    )
    auto_load: Optional[bool] = field(
        default=False
    )

    use_focal_loss: Optional[bool] = field(
        default=False
    )
    focal_loss_gamma: Optional[float] = field(
        default=1
    )

@dataclass
class DataArguments:
    train_data: str = field(
        default='alpaca', metadata={"help": "Name of the training data."}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    cache_dataset: bool = field(
        default=True
    )
    prompt_template_name: str = field(
        default='alpaca'
    )
    val_dataset_size : int = field(
        default=2000
    )
    extra_val_set : str = field(
        default='wikitext2'
    )
    extra_val_seqlen: int = field(
        default=128
    )
    extra_trainset: bool = field(
        default=False
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    optim: str = field(
        default="adamw_hf"
    )
    train_full_prompt: bool = field(
        default=False
    )
    

@dataclass
class SparseArguments:
    # sparse training
    is_sparse_training: Optional[bool] = field(
        default = False
    )
    coef: Optional[float] = field(
        default = 0
    )
    lr_multi: Optional[float] = field(
        default = 100
    )

    is_mask_training: Optional[bool] = field(
        default = False
    )
    mask_config_dir: Optional[str] = field(
        default=None
    )

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SparseArguments)
    )
    model_args, data_args, training_args, sparse_args = parser.parse_args_into_dataclasses()
        
    # from transformers import LlamaForCausalLM, LlamaTokenizer
    from models.llama import LlamaForCausalLM, LlamaTokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    if model_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=model_args.peft_target_modules.split(','),
            inference_mode=False,
            r=model_args.peft_lora_r,
            lora_alpha=model_args.peft_lora_alpha,
            lora_dropout=model_args.peft_lora_dropout,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    train_data, test_data = get_loaders(
        data_args.data_path,
        data_args.val_dataset_size,
        data_args.max_length,
        tokenizer,
        training_args.train_full_prompt,
        data_args.cache_dataset,
        data_args.prompt_template_name
    )
    
    # Load extra validation dataset
    if data_args.extra_val_set:

        def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
            test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
            nsamples = test_ids.numel() // seq_len

            test_set = []
            for i in range(nsamples):
                batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
                test_set.append({
                    'input_ids': batch,
                    'labels': batch
                })
            return test_set
        
        for extra_dataset in data_args.extra_val_set.split(','):
            if 'wikitext2' in extra_dataset:
                extra_test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                extra_test_data = split_and_tokenizer(
                    extra_test_data, tokenizer, 
                    data_args.extra_val_seqlen, 
                    field_name='text'
                )
                test_data[extra_dataset] = extra_test_data

    # pad to longest
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, 
        padding=True,
    )

    trainer = LLaMAMaskTrainer(
        model=model, 
        train_dataset=train_data,
        eval_dataset=test_data,
        args=training_args, 
        sp_args=sparse_args,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model()

    if os.environ.get('LOCAL_RANK', '0') == '0':
        # fuse lora ckpt and save model
        # sparse_mask is bool mask and is ignored without impact
        if model_args.use_lora:
            model = model.merge_and_unload()  # fuse lora ckpt
            state_dict = model.state_dict()
            lora_save_dir = os.path.join(training_args.output_dir, 'lora_fused_model')
            model.save_pretrained(save_directory = lora_save_dir, state_dict=state_dict, max_shard_size="10GB")
            tokenizer.save_pretrained(lora_save_dir)

        ppl_metric(model.half().cuda(), tokenizer, ['wikitext2'], seq_len=128, batch_size=8)

if __name__ == "__main__":
    train()
