depth=16
sparsity=0.4
num_samples=1000
prune_method=llm
epoch="1"
output_name="real_d${depth}_${sparsity}sparsity_${num_samples}i_${prune_method}_method_${epoch}epoch_qfix"
var_model="/home/project/real_prune/VAR_train/qscaluefix_var_d16_0.4_cat680_1epoch/ar-ckpt-last.pth"
CUDA_VISIBLE_DEVICES=2 python FID_test.py --depth $depth --sparsity $sparsity --var_model=$var_model --output_name=$output_name

# var_model="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/d24_0.2var_${num_samples}i_256input_temporary.pth"

# ============================================
# 自动计算FID分数
# ============================================
echo ""
echo "=========================================="
echo "开始计算FID分数..."
echo "=========================================="

# 构建生成的npz文件路径
GENERATED_NPZ="/home/project/real_prune/VAR_FIDtest/output/${output_name}.npz"
REFERENCE_NPZ="/home/project/daily/AR/model_zoo/VIRTUAL_imagenet256_labeled.npz"

# 检查生成的npz文件是否存在
if [ ! -f "$GENERATED_NPZ" ]; then
    echo "错误: 生成的npz文件不存在: $GENERATED_NPZ"
    exit 1
fi

# 检查参考npz文件是否存在
if [ ! -f "$REFERENCE_NPZ" ]; then
    echo "错误: 参考npz文件不存在: $REFERENCE_NPZ"
    exit 1
fi

echo "参考文件: $REFERENCE_NPZ"
echo "生成文件: $GENERATED_NPZ"
echo ""

# 调用evaluator.py计算FID
CUDA_VISIBLE_DEVICES="" python evaluator.py "$REFERENCE_NPZ" "$GENERATED_NPZ"

# CUDA_VISIBLE_DEVICES="" python evaluator.py "/home/project/daily/AR/model_zoo/VIRTUAL_imagenet256_labeled.npz" "/home/project/real_prune/VAR_FIDtest/output/real_d16_0.4sparsity_100i_cat680-fix_method_0epoch.npz"

# 检查evaluator.py是否成功执行
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "FID计算完成!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "FID计算失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi