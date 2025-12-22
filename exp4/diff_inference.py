import argparse
import os
import torch
import torchvision.utils as vutils
from multiprocessing import freeze_support
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.fid import fid_calculator
from utils.inception_score import InceptionScoreCalculator
import torch
from torchvision import datasets, transforms  # 直接导入datasets和transforms
from utils.unet import SimpleUNet
import numpy as np


# --- Pickle Fix: 定义与 diff_train.py 中一致的 Config 类 ---
class Config:
    # 核心训练参数
    batch_size = 256         # input batch size for training
    epochs = 500             # number of epochs to train
    lr = 0.0002              # learning rate
    beta1 = 0.5              # Adam beta1
    
    # Diffusion 参数
    num_timesteps = 1000     # number of diffusion timesteps
    beta_start = 0.0001      # start value of beta for noise schedule
    beta_end = 0.02          # end value of beta for noise schedule

    # 系统与保存参数
    no_cuda = False          # disables CUDA training
    seed = 42                # random seed
    save_model = True        # For Saving the current Model
    load_model = './check_point/best_diffusion_model.pth'       # Path to load model for resuming training
    save_interval = 10       # Interval to save model checkpoints

class DiffInferenceConfig:
    # 路径与文件
    model_path = './check_point/best_diffusion_model.pth'    # 训练好的模型路径
    output_dir = './output_diff'                 # 输出文件夹
    output_name = 'generated_images.png'         # 输出文件名
    
    # 生成参数
    num_images = 64                              # 生成数量
    num_timesteps = 1000                         # 扩散步数 (默认，会被checkpoint覆盖)
    beta_start = 0.0001
    beta_end = 0.02
    
    # 系统参数
    seed = 42
    no_cuda = False

args = DiffInferenceConfig()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 设置设备
device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")


torch.manual_seed(args.seed)

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Initialize the SimpleUNet model
model = SimpleUNet().to(device)

# Load the trained model
if os.path.isfile(args.model_path):
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model'])
    
    if 'args' in checkpoint:
        if hasattr(checkpoint['args'], 'num_timesteps'):
            args.num_timesteps = checkpoint['args'].num_timesteps
        if hasattr(checkpoint['args'], 'beta_start'):
            args.beta_start = checkpoint['args'].beta_start
        if hasattr(checkpoint['args'], 'beta_end'):
            args.beta_end = checkpoint['args'].beta_end
    
    print(f"Model loaded successfully! Using {args.num_timesteps} timesteps.")
else:
    print(f"Error: No model found at {args.model_path}")
    exit(1)

# Set the model to evaluation mode
model.eval()

# Define noise schedule
betas = torch.linspace(args.beta_start, args.beta_end, args.num_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])
alpha_cumprod_next = torch.cat([alphas_cumprod[1:], torch.ones(1, device=device)])

# Precompute required values for denoising
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

# Calculate posterior variance
algorithm_type = "ddpm"
if algorithm_type == "ddpm":
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
else:
    posterior_variance = betas
posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

def p_sample_loop(shape):
    print(f"Generating {shape[0]} images from noise...")
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, args.num_timesteps)), desc="Denoising", total=args.num_timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            noise_pred = model(x, t)
            
            beta_t = betas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha_t = sqrt_recip_alphas[i]
            
            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
            
            # 加入噪声 (除了最后一步 t=0)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t) # Option 1: sigma = sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean # 最后一步不加噪声
                
    return x

def p_sample(model, x, t):
    """Sample x_{t-1} from x_t and model"""
    # Get current timestep values
    sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t][:, None, None, None]
    sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t][:, None, None, None]
    posterior_variance_t = posterior_variance[t][:, None, None, None]
    posterior_mean_coef1_t = posterior_mean_coef1[t][:, None, None, None]
    posterior_mean_coef2_t = posterior_mean_coef2[t][:, None, None, None]
    
    # Predict noise
    noise_pred = model(x, t)
    
    # Calculate mean and variance for the reverse process
    mean = sqrt_recip_alphas_cumprod_t * (x - sqrt_recipm1_alphas_cumprod_t * noise_pred)
    
    # If we're at t=0, return the mean (no more noise)
    if t[0] == 0:
        return mean
    else:
        # Add noise to the mean according to the posterior variance
        noise = torch.randn_like(x)
        variance = torch.exp(0.5 * posterior_log_variance_clipped[t])[:, None, None, None]
        return mean + variance * noise

# Generate images

generated_images_list = []
for i in tqdm(range(0, args.num_images, 32), desc="Generating"):
    current_batch_size = min(32, args.num_images - i)
    noise_shape = (current_batch_size, 3, 32, 32)
    generated_images_batch = p_sample_loop(noise_shape)
    generated_images_list.append(generated_images_batch)
generated_images = torch.cat(generated_images_list, dim=0)

print("Calculating FID...")
random.seed(args.seed)
real_indices = random.sample(range(len(test_dataset)), args.num_images)
real_images = torch.stack([test_dataset[i][0] for i in real_indices]).to(device)
    
# Calculate FID score
fid_score = fid_calculator(real_images, generated_images, batch_size=32, device=device)
print(f"FID Score: {fid_score:.4f}")

print("Calculating Inception Score...")
is_calculator = InceptionScoreCalculator(device=device)
inception_score, std_inception_score = is_calculator.compute_inception_score(
    generated_images
)
print(f"Inception Score: {inception_score:.4f} ± {std_inception_score:.4f}")

generated_images_save = generated_images[:64]
output_path = os.path.join(args.output_dir, args.output_name)
vutils.save_image(generated_images_save, output_path, normalize=True, nrow=8)

print(f"Generated {args.num_images} images and saved to {output_path}")

print("\nDisplaying generated results:")
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title(f"Seed: {args.seed} | File: {args.output_name}")
grid_img = vutils.make_grid(generated_images_save.cpu()[:64], padding=2, normalize=True, nrow=8)
plt.imshow(np.transpose(grid_img, (1, 2, 0)))
plt.show()
