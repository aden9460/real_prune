# import numpy as np
# from PIL import Image
# import os

# # data = np.load('/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/VIRTUAL_imagenet256_labeled.npz')
# # print(data['arr_0'].shape)
# # print(data['arr_0'].dtype)
# # arr = np.load('/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/VIRTUAL_imagenet256_labeled.npz')['images']
# # save_dir = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images'
# # os.makedirs(save_dir, exist_ok=True)
# # for i, img in enumerate(arr):
# #     Image.fromarray(img).save(os.path.join(save_dir, f'{i:06d}.png'))


# # 加载 npz 文件
# data = np.load('/home/wangzefang/Projects/EdgeVAR/LlamaGen/autoregressive/model_zoo/VIRTUAL_imagenet256_labeled.npz')
# print(data.files)
# images = data['arr_0']

# # 保存目录
# save_dir = '/home/wangzefang/Projects/EdgeVAR/LlamaGen/autoregressive/virtual_images/'
# os.makedirs(save_dir, exist_ok=True)

# # 循环保存图片
# for i, img in enumerate(images):
#     Image.fromarray(img).save(os.path.join(save_dir, f'{i:06d}.png'))

# print(f"已保存 {len(images)} 张图片到 {save_dir}")

import numpy as np
from PIL import Image
import os

data = np.load('/home/wangzefang/Projects/EdgeVAR/LlamaGen/autoregressive/model_zoo/VIRTUAL_imagenet256_labeled.npz')
print(data.files)

save_dir = '/home/wangzefang/Projects/EdgeVAR/LlamaGen/autoregressive/virtual_images/'
os.makedirs(save_dir, exist_ok=True)

for k in data.files:
    arr = data[k]
    if k == 'arr_0':
        # 保存图片
        img_dir = os.path.join(save_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(arr):
            Image.fromarray(img).save(os.path.join(img_dir, f'{i:06d}.png'))
        print(f"已保存 {len(arr)} 张图片到 {img_dir}")
    else:
        # 保存统计量为 .npy
        np.save(os.path.join(save_dir, f'{k}.npy'), arr)
        print(f"已保存 {k}.npy 到 {save_dir}")