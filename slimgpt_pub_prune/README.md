# SlimGPT: Layer-wise Structured Pruning for Large Language Models

With just one GPU card, ten to thirty minutes, and a few hundred pre-trained data points, we can quickly perform structured pruning on a large language model.

## Features

- Low-cost, rapid pruning. A 7B model can be pruned by 20% in just 10 minutes.
- Supports two ways of fine-tuning:
  - Fine-tuning with sparse mask, keeping the pruned parameters from updating.
  - Fine-tuning with pruned models, directly loading the smaller model after pruning for fine-tuning.
- Supports saving the checkpoint after pruning, which has a smaller number of parameters.
- By slightly modifying the model description file, the pruned model supports the general Hugging Face interface.

## Directory Tree
```txt
.
├── README.md
├── configs
│   └── deepspeed.json
├── lm-evaluation-harness
├── model_evaluate.py
├── model_finetune.py
├── model_slimming.py
├── models
│   └── llama
├── ppl_eval
│   ├── ppl_dataset.py
│   └── ppl_eval.py
├── requirements.txt
├── scripts
├── slim_utils
│   ├── params_remove.py
│   ├── slim_dataset.py
│   └── slimgpt.py
├── test_flops.py
└── train_utils
    ├── finetune_dataset.py
    └── sparse_trainer.py
```

## Getting Started

**Environment setup**
```sh
cd SlimGPT
/opt/conda/envs/python3.8/bin/pip install -r requirements.txt
```

**Prepare Model**

Download the LLaMA-7B model from Modelscope, where you can freely choose the download source.

```python
/opt/conda/envs/python3.8/bin/python 
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('skyline2006/llama-7b', cache_dir='./tmp_model_name_or_path')
```

**Prepare Dataset**

If you encounter any issues with downloading the data, or if you prefer to load the data locally, you can first download the data and then modify the loading method at the corresponding location in `slim_utils/slim_datset.py`.

- For downloading the c4 data, go [here](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz) to download the data and save it locally. Afterwards, in `slim_utils.slim_datset.get_c4`, set the local path with `data_file = /YOUR/DATASET/PATH`.
- For downloading the alpaca data, go [here](https://huggingface.co/datasets/yahma/alpaca-cleaned/blob/main/alpaca_data_cleaned.json) to download the data and save it locally. Then, in `slim_utils.slim_datset.get_alpaca`, set the local path with `data_file = /YOUR/DATASET/PATH`.
- For downloading the gpt4_alpaca data, go [here](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) to download the data and save it locally. Next, in `slim_utils.slim_datset.get_alpaca`, set the local path with `data_file = /YOUR/DATASET/PATH`.

### Step1: Pruning with SlimGPT

```sh
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/python3.8/bin/python model_slimming.py \
./tmp_model_name_or_path/skyline2006/llama-7b \
c4 \
--minlayer 0 \
--maxlayer 32 \
--num_samples 128 \
--seqlen 2048 \
--percdamp 0 \
--non_uniform \
--non_uniform_strategy log_increase \
--min_sparsity 0.0625 \
--max_sparsity 0.25 \
--save_pruned_weights \
--save_dir ./tmp_model_name_or_path/llama_7b_p20 2>&1 | tee -i ./tmp_model_name_or_path/llama_7b_p20.ppl
```
Now we have a model weights with certain columns reset to zero and a pruned model checkpoint.

**NOTE:** replace `/opt/conda/envs/python3.8/bin/python` to your conda environment path.

> If the HF dataset fails to load, try: `export HF_ENDPOINT=https://hf-mirror.com`

### Step2: Finetune with LORA (Optional)

```sh
CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/python3.8/bin/torchrun --master_port=65535 --master_addr=127.0.0.1 --nproc_per_node=2 --nnodes=1 --node_rank=0 model_finetune.py \
--data_path /tao-m6-gpt/chenhai/datasets/alpaca-cleaned/alpaca_data_cleaned.json \
--deepspeed configs/deepspeed.json \
--bf16 true \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--num_train_epochs 1 \
--lr_scheduler_type cosine \
--learning_rate 1e-4 \
--warmup_steps 10 \
--logging_steps 50 \
--logging_first_step true \
--evaluation_strategy steps \
--eval_steps 100 \
--max_length 256 \
--cache_dataset false \
--optim adamw_hf \
--use_lora true \
--is_mask_training false \
--model_name_or_path ./tmp_model_name_or_path/llama_7b_p20/pruned_weights \
--output_dir ./tmp_model_name_or_path/llama_7b_sft 2>&1 | tee -i ./tmp_model_name_or_path/llama_7b_sft.log 
```

### Step3: Evaluate Pruned Model

We use modified lm-evaluation-harness for evaluating.

```sh
cd lm-evaluation-harness
/opt/conda/envs/python3.8/bin/pip install -e .
cd ..
```

Evaluate pruned model.

```sh
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/python3.8/bin/python model_evaluate.py \
--model ./tmp_model_name_or_path/llama_7b_sft/lora_fused_model \
--tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
--device cuda:0 \
--batch_size 32 \
--output_path ./tmp_model_name_or_path/llama_7b_sft.json \
--no_cache
```

## Reference

- [LLM-Pruner](https://github.com/horseee/LLM-Pruner)
- [gptq](https://github.com/ist-daslab/gptq)
- [sparsegpt](https://github.com/ist-daslab/sparsegpt)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/)
