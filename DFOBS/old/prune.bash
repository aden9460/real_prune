specific_layer=256
maxlayer=16 
sparsity=0.4
num_samples=150
model_name="prune_d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale.pth" 

CUDA_VISIBLE_DEVICES=2 python -u prune.py \
  --minlayer 0 \
  --maxlayer $maxlayer \
  --num_samples $num_samples \
  --percdamp 1e-2 \
  --prune_method slimgpt \
  --sparsity $sparsity \
  --specific_layer $specific_layer \
  --model_name $model_name

  # --seqlen $seqlen \
  # --min_sparsity 0.0625 \
  # --max_sparsity 0.3 \
  # --non_uniform \
  # --non_uniform_strategy linear_decrease \