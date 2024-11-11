import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 加载保存的张量
save_dir = "cls_debug_outputs"
# gate_score = torch.load(os.path.join(save_dir, "gate_score.pt"))
expert_outputs_before_sr = torch.load(
    os.path.join(save_dir, "expert_outputs_before_sr.pt")
)
expert_output = torch.load(os.path.join(save_dir, "expert_output.pt"))
expert_output = [i.view(1, 14, 14, -1) for i in expert_output]

expert_outputs_after_sr = torch.load(
    os.path.join(save_dir, "expert_output_after_sr.pt")
)
expert_outputs_after_sr = [i.view(1, 14, 14, -1) for i in expert_outputs_after_sr]

output = torch.load(os.path.join(save_dir, "output.pt"))
output = [i.view(1, 14, 14, -1) for i in output]


# 计算正方形边长
def calculate_square_side(hw):
    side = int(np.sqrt(hw))
    if side * side != hw:
        raise ValueError(
            f"hw ({hw}) is not a perfect square, cannot form a square image."
        )
    return side


# 可视化函数
def visualize_gate_score(gate_scores, save_dir):
    for idx, score in enumerate(gate_scores):
        b, e, h, w = score.shape
        fig, axes = plt.subplots(nrows=1, ncols=e, figsize=(15, 5))
        for i in range(e):
            ax = axes[i]
            cax = ax.imshow(
                score[0, i].detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=1
            )
            ax.set_title(f"gate_score layer {idx+1} expert {i+1}")
            ax.set_xlabel("width")
            ax.set_ylabel("height")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"gate_score_layer_{idx+1}.png"))
        # plt.show()
        plt.close(fig)


def visualize_expert_outputs(expert_outputs, save_dir, prefix):
    for idx, outputs in enumerate(expert_outputs):
        b, hw, c = outputs[0].shape
        side = calculate_square_side(hw)
        all_values = [
            output[0].view(side, side, c).permute(2, 0, 1)[0] for output in outputs
        ]
        all_values = torch.cat(all_values).detach().cpu().numpy()
        vmin, vmax = all_values.min(), all_values.max()

        fig, axes = plt.subplots(nrows=1, ncols=len(outputs), figsize=(15, 5))
        for i, output in enumerate(outputs):
            b, hw, c = output.shape
            side = calculate_square_side(hw)
            attention_map = (
                output[0].view(side, side, c).permute(2, 0, 1).detach().cpu().numpy()[0]
            )
            ax = axes[i]
            cax = ax.imshow(attention_map, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{prefix} layer {idx+1} expert {i+1}")
            ax.set_xlabel("width")
            ax.set_ylabel("height")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_layer_{idx+1}.png"))
        # plt.show()
        plt.close(fig)


def visualize_outputs(outputs, save_dir, prefix):
    for idx, output in enumerate(outputs):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        attention_map = output[0].detach().cpu().numpy()[:,:,0]
        cax = ax.imshow(attention_map, cmap="viridis")
        ax.set_title(f"{prefix} layer {idx+1}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_layer_{idx+1}.png"))
        # plt.show()
        plt.close(fig)


def concatenate_images(image_dir, output_dir, image_prefix, image_suffix):
    """
    将同类型的图片按照顺序从上到下拼接。

    :param image_dir: 包含图片的目录路径
    :param output_dir: 输出拼接后图片的目录路径
    :param image_prefix: 图片文件名的前缀
    :param image_suffix: 图片文件名的后缀
    """
    # 获取所有符合条件的图片文件名
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.startswith(image_prefix) and f.endswith(image_suffix)
    ]

    def extract_number(filename):
        """从文件名中提取数字，如果找不到数字则返回0"""
        match = re.search(r"_(\d+)\.", filename)  # 修改正则表达式以匹配下划线后的数字
        return int(match.group(1)) if match else 0

    image_files.sort(key=extract_number)  # 按文件名排序

    if not image_files:
        print(
            f"No images found with prefix '{image_prefix}' and suffix '{image_suffix}' in directory '{image_dir}'."
        )
        return

    # 读取第一张图片以获取宽度和高度
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = Image.open(first_image_path)
    width, height = first_image.size

    # 计算总高度
    total_height = len(image_files) * height

    # 创建一个新的空白图像
    new_image = Image.new("RGB", (width, total_height))

    # 将所有图片拼接到新图像中
    y_offset = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        new_image.paste(image, (0, y_offset))
        y_offset += height

    # 保存拼接后的图像
    output_filename = f"concatenated_{image_prefix[:-1]}.png"
    output_path = os.path.join(output_dir, output_filename)
    new_image.save(output_path)

    print(f"Concatenated image saved to {output_path}")


# 可视化 gate_score
# visualize_gate_score(gate_score, save_dir)

# 可视化 expert_outputs_before_sr
visualize_expert_outputs(expert_outputs_before_sr, save_dir, "wkv")

visualize_outputs(expert_output, save_dir, "wkv_mean")

# 可视化 expert_outputs_after_sr
visualize_outputs(expert_outputs_after_sr, save_dir, "rwkv_mean")

visualize_outputs(output, save_dir, "output")

# 拼接三种类型的图片
concatenate_images(save_dir, save_dir, "wkv_layer_", ".png")
concatenate_images(save_dir, save_dir, "wkv_mean_layer_", ".png")
concatenate_images(save_dir, save_dir, "rwkv_mean_layer_", ".png")
concatenate_images(save_dir, save_dir, "output_layer_", ".png")
