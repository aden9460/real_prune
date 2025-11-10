depth=16
sparsity=0.4
num_samples=150
prune_method=prowithoutwaare
epoch="20+20"
output_name="real_d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method_${epoch}epoch"
var_model="/home/project/real_prune/VAR_train/train_result/fix_distill_d16_normal_256_from20_progresswithoutsaware/ar-ckpt-last.pth"
CUDA_VISIBLE_DEVICES=0 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"