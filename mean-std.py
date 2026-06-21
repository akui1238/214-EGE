import os
from PIL import Image
import numpy as np

# 指定图像文件夹路径
# image_folder_path = r"D:\edge-download\EGE-UNet-eye\EGE-UNet-eye\data\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\images"
image_folder_path = r"D:\edge-download\EGE-UNet-eye\EGE-UNet-eye\data\HRF\train\images"
# 初始化一个空列表来存储所有图像的数据
all_images = []

# 遍历文件夹中的所有图像文件
for filename in os.listdir(image_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据实际情况调整图像格式
        # 构建完整的文件路径
        file_path = os.path.join(image_folder_path, filename)
        # 打开图像并转换为NumPy数组
        img = Image.open(file_path)
        img_np = np.array(img)
        # 将图像数据添加到列表中
        all_images.append(img_np)

# 将所有图像数据堆叠成一个四维数组
all_images_np = np.stack(all_images, axis=0)

# 计算每个颜色通道的平均值
mean = np.mean(all_images_np, axis=(0, 1, 2))

# 计算每个颜色通道的标准差
std = np.std(all_images_np, axis=(0, 1, 2))

# 输出每个颜色通道的平均值和标准差
print(f"Mean per channel: {mean}")
print(f"Standard Deviation per channel: {std}")
