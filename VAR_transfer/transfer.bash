torchrun  --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=29501 transfer_var.py \
  --depth=24 --transfer=True --sparsity=0 --data_path="/home/suanba/datasets/ImageNet-1K/" 