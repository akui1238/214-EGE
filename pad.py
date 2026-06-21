# import os
# from PIL import Image
#
# def pad_and_save_image(image_path, output_dir, output_size=(704, 608)):
#     # 定义一个名为 pad_and_save_image 的函数，它接受三个参数：image_path（图像文件的路径），output_dir（输出目录的路径），
#     # 以及一个可选参数 output_size（输出图像的大小，默认为 (704, 608)）。
#
#     # Open the image
#     # image = Image.open(image_path)
#     image = Image.open(r"D:\edge - download\STARE - seg\train\images1")
#
#
#     # Calculate the padding needed
#     current_size = image.size
#     padding_width = output_size[0] - current_size[0]
#     padding_height = output_size[1] - current_size[1]
#
#     # Pad the image
#     padded_image = Image.new('RGB', output_size)
#     padded_image.paste(image, (padding_width // 2, padding_height // 2))
#
#     # Save the padded image
#     # output_path = os.path.join(output_dir, os.path.basename(image_path))
#     output_path = os.path.join(r"D:\edge - download\STARE - seg\train\images", os.path.basename(image_path))
#     padded_image.save(output_path)
#
#     return output_path

import os
from PIL import Image

def pad_images_in_folder(image_folder, output_folder, output_size):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.ppm', '.tif')):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing image: {image_path}")
            # 打开图像
            image = Image.open(image_path)
            # 获取原始图像尺寸
            current_size = image.size
            # 计算填充尺寸
            padding_width = max(output_size[0] - current_size[0], 0)
            padding_height = max(output_size[1] - current_size[1], 0)
            # 创建新的图像
            padded_image = Image.new('RGB', output_size, (255, 255, 255))
            # 粘贴原始图像到新图像
            padded_image.paste(image, (padding_width // 2, padding_height // 2))
            # 保存填充后的图像
            output_path = os.path.join(output_folder, filename)
            padded_image.save(output_path)
            print(f"Saved padded image: {output_path}")

# 使用示例
image_folder = r"/home/ydm/EGE-UNet-eye/data/DRIVE/train/images"
output_folder = r"/home/ydm/EGE-UNet-eye/data/DRIVE/train/images"
output_size = (576,608)  # 指定输出图像的尺寸
pad_images_in_folder(image_folder, output_folder, output_size)

