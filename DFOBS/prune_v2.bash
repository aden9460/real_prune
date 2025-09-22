specific_layer=256
maxlayer=16
sparsity=0.4
num_samples=10
prune_method=woodtaylor
model_name="real_d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale_${prune_method}.pth" 

CUDA_VISIBLE_DEVICES=0 python -u DF_v1.py\
  --minlayer 0 \
  --maxlayer $maxlayer \
  --num_samples $num_samples \
  --percdamp 1e-2 \
  --skip_evaluate \
  --prune_method $prune_method \
  --sparsity $sparsity \
  --specific_layer $specific_layer \
  --model_name $model_name \
  --H_mode 'fisher' \
  --save_dir "/home/suanba/real_prune/DFOBS/output" \
  --fisher_batches $num_samples \
  # --seqlen $seqlen \
  # --min_sparsity 0.0625 \
  # --max_sparsity 0.3 \
  # --non_uniform \
  # --non_uniform_strategy linear_decrease \