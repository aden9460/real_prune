specific_layer=256
maxlayer=16
sparsity=0.4
num_samples=10
prune_method=woodtaylor
H_mode=fisher
fisher_loss_scope="global"
model_name="v4_new_real_${fisher_loss_scope}_d${maxlayer}_${sparsity}sparsity_${num_samples}_i_${specific_layer}eva_scale_${prune_method}_${H_mode}.pth" 


CUDA_VISIBLE_DEVICES=0 python -u DF_v4.py\
  --minlayer 0 \
  --maxlayer $maxlayer \
  --model_depth $maxlayer \
  --num_samples $num_samples \
  --percdamp 1e-2 \
  --skip_evaluate \
  --prune_method $prune_method \
  --sparsity $sparsity \
  --specific_layer $specific_layer \
  --model_name $model_name \
  --H_mode $H_mode \
  --save_dir "/home/waas/real_prune/DFOBS/output" \
  --fisher_batches $num_samples \
  --fisher_loss_scope $fisher_loss_scope \
  # --use_fisher_sparsity  \
  # --seqlen $seqlen \
  # --min_sparsity 0.0625 \
  # --max_sparsity 0.3 \
  # --non_uniform \
  # --non_uniform_strategy linear_decrease \