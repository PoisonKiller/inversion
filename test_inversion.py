import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler
from inverse_pipeline import StableDiffusionDiffEditPipeline
from torchvision.utils import save_image
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest
import os 

device='cuda:5'
# 初始化管道
pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "/home/cjt/pretrained_models/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
    local_file_only=True
).to(device)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
# pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

def invert_image(pipe, image_path, resolution, save_path):
        # 加载原始图像
    img_url = f"/home/cjt/Kodak/kodim0{i}.png"
    raw_image = load_image(img_url).resize((512, 512))
    # raw_image = load_image(img_url)

    # 生成 mask 图像
    source_prompt = ""
    target_prompt = ""
    mask_image = pipe.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    mask_image = mask_image * 0.

    # 转换 mask 为 PIL 图像并保存
    mask_image_pil = Image.fromarray((mask_image.squeeze() * 255).astype("uint8"), "L").resize((512, 512))
    # mask_image_pil = Image.fromarray((mask_image.squeeze() * 255).astype("uint8"), "L")
    # mask_image_pil.save("mask_image.png")
    # 反向推理获取潜变量
    inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents

    # 生成最终图像
    output_image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        negative_prompt=source_prompt,
    ).images[0]

    # 保存输出图像
    output_image.save("output_image.png")

    # # 生成对比网格并保存
    comparison_grid = make_image_grid([raw_image, output_image], rows=1, cols=2)
    comparison_grid.save(os.path.join(save_path, "comparison_grid0{i}.png"))

for i in range(1, 10):
    # 加载原始图像
    img_url = f"/home/cjt/Kodak/kodim0{i}.png"
    invert_image


# def plot_latent_gray(latents, out_dir):
#     tensor = latents[0][35].detach().cpu()  # 去除 batch 维度并 detach
#     print(tensor.shape)

#     # 创建 1x4 子图
#     fig, axes = plt.subplots(1, 4, figsize=(24, 4))

#     for i in range(tensor.size(0)):  # 遍历 4 个通道
#         channel_data = tensor[i]
        
#         # 归一化到 [0, 255]
#         channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
#         channel_data = (channel_data * 255).type(torch.uint8).numpy()
        
#         # 在对应的子图上显示
#         axes[i].imshow(channel_data, cmap='gray')
#         axes[i].set_title(f"Channel {i+1}")
#         axes[i].axis('off')

#     plt.tight_layout()
    
#     # 保存图像
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     save_path = os.path.join(out_dir, "latent_channels35.png")
#     plt.savefig(save_path)
#     plt.show()
    
#     print(f"Saved latent visualization to {save_path}")

# plot_latent_gray(inv_latents, "./")

# 假设 inv_latents 是一个 torch.Tensor
# inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents

# 转换为 NumPy 数组
# inv_latents_np = inv_latents.cpu().numpy()

# # 展开为一维数组
# flat_inv_latents = inv_latents_np.flatten()

# # 1. 计算均值和标准差
# mean = np.mean(flat_inv_latents)
# std = np.std(flat_inv_latents)
# print(f"Mean: {mean}, Std: {std}")

# # 2. 绘制直方图
# plt.hist(flat_inv_latents, bins=100, density=True, alpha=0.6, color='g')
# plt.title("Histogram of inv_latents")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.savefig('distributions.jpg')

# # 3. 正态性检验
# # Shapiro-Wilk Test
# shapiro_test = shapiro(flat_inv_latents[:5000])  # 取前5000个数据避免内存问题
# print(f"Shapiro-Wilk Test: W={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# # Kolmogorov-Smirnov Test
# ks_test = kstest(flat_inv_latents, 'norm', args=(mean, std))
# print(f"Kolmogorov-Smirnov Test: D={ks_test.statistic}, p-value={ks_test.pvalue}")