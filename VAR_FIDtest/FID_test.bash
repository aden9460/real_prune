depth=16
sparsity=0.2
num_samples=150
prune_method="d16_0.2_new_average_1epoch"
output_name="real_d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method"
var_model="/home/suanba/EdgeVAR/real_prune/VAR_train/0.2_d16_real_1epoch_new_average_slimgpt/ar-ckpt-last.pth"
CUDA_VISIBLE_DEVICES=0 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"