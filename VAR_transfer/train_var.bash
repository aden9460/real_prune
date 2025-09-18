# echo "==> Training..."


# #!/bin/sh 
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#         . "/opt/conda/etc/profile.d/conda.sh"
#     else
#         export PATH="/opt/conda/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate slim

# # 训练参数
# TRAIN_SCRIPT="train.py"
# TRAIN_ARGS="--depth=24 --bs=240 --ep=20 --fp16=1 --sparsity=0.2 --local_out_dir_path="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-20" --data_path="/wanghuan/data/wangzefang/ImageNet-1K/""

# # 配置弹性参数
# NNODES="1:4"
# NPROC_PER_NODE=6
# MAX_RESTARTS=100

# # 用平台自动注入的job id作为rdzv_id，确保所有节点一样
# RDZV_ID=${VC_JOB_ID:-myjob20240513}
# RDZV_BACKEND="c10d"
# RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# # 打印环境变量，方便排查
# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "MASTER_PORT=$MASTER_PORT"
# echo "RDZV_ID=$RDZV_ID"

# torchrun \
#   --nnodes=$NNODES \
#   --nproc_per_node=$NPROC_PER_NODE \
#   --max_restarts=$MAX_RESTARTS \
#   --rdzv_id=$RDZV_ID \
#   --rdzv_backend=$RDZV_BACKEND \
#   --rdzv_endpoint=$RDZV_ENDPOINT \
#   $TRAIN_SCRIPT $TRAIN_ARGS





# CUDA_VISIBLE_DEVICES=3,4,5,6,7 

# torchrun  \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   --node_rank=0 \
#   train.py \
#   --depth=16 --bs=384 --ep=20 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/home/suanba/EdgeVAR/real_prune/VAR_train/0.2_d16_real_20epoch_8_384_maginitude" --data_path="/home/suanba/datasets/ImageNet-1K" \
#   --var_path="/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/sparsity_model/real_d16_0.2sparsity_1i_256eva_scale_magnitude.pth" \
#   --vae_path='/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth'



#!/bin/bash

# 要执行的 Python 程序路径
PYTHON_SCRIPT="/path/to/your_script.py"

while true; do
    # 第一次检测 GPU7 显存占用（MiB）
    MEM1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)

    if [ "$MEM1" -lt 10 ]; then
        echo "GPU7 第一次检测显存占用为 0 MiB，3 秒后再次确认..."
        sleep 3

        # 第二次检测
        MEM2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)

        if [ "$MEM2" -lt 10 ]; then
            echo "GPU7 连续两次显存占用为 0 MiB，执行 Python 程序..."
            CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun  \
            --nnodes=1 \
            --nproc_per_node=5 \
            --node_rank=0 \
            train.py \
            --depth=16 --bs=320 --ep=1 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/home/suanba/EdgeVAR/real_prune/VAR_train/0.2_d16_real_1epoch_5_384_taylor" --data_path="/home/suanba/datasets/ImageNet-1K" \
            --var_path="/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/sparsity_model/real_d16_0.2sparsity_1i_256eva_scale_taylor.pth" \
            --vae_path='/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth'


            break
        else
            echo "GPU7 第二次检测占用为 ${MEM2} MiB，继续等待..."
        fi
    else
        echo "GPU7 第一次检测占用为 ${MEM1} MiB，等待中..."
    fi

    # 每 10 秒检测一次
    sleep 500
done


# torchrun \
#   --nproc_per_node=6 \
#   --nnodes=2 \
#   --node_rank=1 \
#  train.py \
#   --depth=24 --bs=252 --ep=10 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=0.2 --local_out_dir_path="/wanghuan/data/wangzefang/slim_VAR_copy/VAR/d20_0.2_0-10" --data_path="/wanghuan/data/wangzefang/ImageNet-1K/"
