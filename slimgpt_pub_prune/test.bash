specific_layer=(9 9 9 8 8 8 7 7 7 6 6 6 5 5 4 4 3 3 2 2 1 1 0 0 )  #test1
# 定义 patch_nums 为可索引的 bash 数组
patch_nums=(1 2 3 4 5 6 8 10 13 16)


scale_values=()
for idx in "${specific_layer[@]}"; do
    scale_values+=("${patch_nums[$idx]}")
done
echo "scale_values: ${scale_values[@]}"


specific_layer_idx=()
for ((i=0; i<24; i++)); do
    specific_layer_idx+=($((RANDOM % 10)))
done
echo "specific_layer_idx: ${specific_layer_idx[@]}"

