import os
import torch
from pathlib import Path
from datasets import load_dataset


class AlpacaPrompter(object):
    def __init__(self, prompt_template_name):
        self.template = self._load_template(prompt_template_name)

    def _load_template(self, prompt_template_name):
        if prompt_template_name.lower() == 'none':
            template = {
                "description": "Template used by Alpaca-LoRA.",
                "prompt_input": "{instruction}\n{input}\n",
                "prompt_no_input": "{instruction}\n",
            }
        elif prompt_template_name.lower() == 'alpaca':
            template = {
                "description": "Template used by Alpaca-LoRA.",
                "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            }
        elif prompt_template_name.lower() == 'alpaca_zh':
            template = {
                "description": "Template used by Alpaca-LoRA.",
                "prompt_input": "下面是一个描述任务的指令与进一步的输入信息，请根据指令提供一个合适的回答。\n\n### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n",
                "prompt_no_input": "下面是一个描述任务的指令，请根据指令提供一个合适的回答。\n\n### 指令:\n{instruction}\n\n### 回答:\n",
            }
        else:
            raise Exception(f'{prompt_template_name} NOT SUPPORTED!')
        return template
    
    def generate_prompt(
        self,
        instruction: str,
        input: str = '',
        output: str = '',
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        res += output
        return res

class PromptTokenizer(object):
    def __init__(self, prompter, tokenizer, max_length, train_full_prompt):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_full_prompt = train_full_prompt

    def generate_and_tokenize_prompt(self, data_point):
        prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"]
        )
        response = data_point["output"]

        # add_bos_token default to True while add_eos_token default to False
        # only truncate, not pad, pad in  DataCollatorForSeq2Seq
        input_ids = self.tokenizer.encode(
            prompt + response,
            add_special_tokens=True,  # add_bos_token
            padding=False, 
            max_length= self.max_length,
            truncation=True
        )
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,  # add_bos_token
            padding=False, 
            max_length= self.max_length,
            truncation=True
        )

        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        labels = input_ids.copy()
        if not self.train_full_prompt:
            labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def get_loaders(
        data_path, 
        val_dataset_size, 
        max_length, 
        tokenizer, 
        train_full_prompt=False, 
        cache_dataset=False, 
        prompt_template_name='alpaca'
    ):

    if cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(data_path)):
        preprocessed_data = torch.load('datasets/cache/{}.bin'.format(data_path))
        train_data, test_data = preprocessed_data['train'], preprocessed_data['test']
    else:
        prompter = AlpacaPrompter(prompt_template_name)
        prompt_tokenizer = PromptTokenizer(prompter, tokenizer, max_length, train_full_prompt)

        assert data_path.endswith('.json')
        data = load_dataset('json', data_files=data_path)
        train_test_split = data["train"].train_test_split(
            test_size=val_dataset_size, shuffle=True, seed=24
        )
        train_data = (
            train_test_split["train"].map(prompt_tokenizer.generate_and_tokenize_prompt)
        )
        test_data = {
            data_path: train_test_split["test"].map(prompt_tokenizer.generate_and_tokenize_prompt),
        }
        if cache_dataset and os.environ.get('LOCAL_RANK', '0') == '0':
            cache_file = 'datasets/cache/{}.bin'.format(data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save(
                {'train': train_data, 'test': test_data},
                cache_file
            )
    return train_data, test_data