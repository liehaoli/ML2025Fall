import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from utils.unet import SimpleUNet

# --- 1. 可视化配置参数 ---
class VizConfig:
    # 路径配置
    model_path = './best_diffusion_model.pth'
    output_dir = './output_diff_steps'  # 🆕 可视化结果保存路径
    
    # 生成参数
    num_images = 64              # 仅需生成 64 张
    
    # 扩散模型参数 (必须与训练一致)
    num_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # 系统参数
    seed = 42
    no_cuda = False

args = VizConfig()

def visualize_diffusion_process():
    # 1. 环境设置
    global device
    if 'device' not in globals():
        device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"Running visualization on: {device}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 2. 准备参数
    betas = torch.linspace(args.beta_start, args.beta_end, args.num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # 3. 加载模型
    try:
        model = SimpleUNet().to(device)
        if os.path.isfile(args.model_path):
            print(f"Loading model from {args.model_path}...")
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {args.model_path}")
            return
    except Exception as e:
        print(f"Error: {e}")
        return

    model.eval()

    # --- 4. 带可视化保存的采样函数 ---
    print(f"Starting visualization process (Total Steps: {args.num_timesteps})...")
    
    with torch.no_grad():
        # 1. 从纯高斯噪声开始 x_T (Step 1000)
        x = torch.randn(args.num_images, 3, 32, 32, device=device)
        
        # 保存初始噪声状态 (Step 1000)
        # 严格来说这还没开始去噪，但为了完整性可以存一下，或者按你的要求只存整百步
        # vutils.save_image(x, f"{args.output_dir}/diffusion_step_noise.png", normalize=True, nrow=8)
        
        # 2. 倒序去噪 T-1 -> 0
        for i in tqdm(reversed(range(0, args.num_timesteps)), desc="Denoising Progress", total=args.num_timesteps):
            t = torch.full((args.num_images,), i, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = model(x, t)
            
            # 提取系数
            beta_t = betas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha_t = sqrt_recip_alphas[i]
            
            # 计算均值 (x_{t-1})
            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean
            
            # --- 🆕 可视化保存逻辑 ---
            # 每 100 步保存一次，或者最后一步保存
            if i % 100 == 0 or i == 0:
                save_path = os.path.join(args.output_dir, f'diffusion_step_{i}.png')
                vutils.save_image(x, save_path, normalize=True, nrow=8)
                # 仅打印关键节点日志，避免刷屏
                if i % 200 == 0 or i == 0:
                    print(f"   📸 Snapshot saved: Step {i}")

    print(f"Visualization completed! Check images in {args.output_dir}")
    
    # --- 5. 展示结果 (展示首尾) ---
    steps_to_show = [500, 300, 100, 0]
    
    print(f"\n Displaying steps: {steps_to_show}")
    
    # 创建 2行 x 3列 的画布
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for idx, step in enumerate(steps_to_show):
        img_path = os.path.join(args.output_dir, f'diffusion_step_{step}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axs[idx].imshow(img)
            axs[idx].set_title(f"Step {step}")
            axs[idx].axis("off")
        else:
            print(f"⚠️ Warning: Image for step {step} not found.")
            axs[idx].axis("off")
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_diffusion_process()
