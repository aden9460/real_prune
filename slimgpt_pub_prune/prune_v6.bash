specific_layer=256
maxlayer=16
sparsity=0.4
num_samples=100
prune_method="slimgpt"
model_name="real_d${maxlayer}_${sparsity}sparsity_${num_samples}i_cat680_scale_${prune_method}_fix_prune.pth" 

CUDA_VISIBLE_DEVICES=0 python -u prune_v6.py\
  --minlayer 0 \
  --maxlayer $maxlayer \
  --num_samples $num_samples \
  --percdamp 1e-2 \
  --skip_evaluate \
  --prune_method $prune_method \
  --sparsity $sparsity \
  --specific_layer $specific_layer \
  --model_name $model_name \

  # --seqlen $seqlen \
  # --min_sparsity 0.0625 \
  # --max_sparsity 0.3 \
  # --non_uniform \
  # --non_uniform_strategy linear_decrease \