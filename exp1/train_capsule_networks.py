import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy

# --- helper: save plots incrementally ---
def save_training_curves(train_losses, train_margin_losses, train_recon_losses,
                         train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    if len(train_losses) == 0:
        return
    # Loss figure (dual axes): recon on left, train/margin on right
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l3, = ax1.plot(epochs, train_recon_losses, label='Reconstruction Loss', color='tab:green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    l1, = ax2.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    l2, = ax2.plot(epochs, train_margin_losses, label='Margin Loss', color='tab:orange')
    ax2.set_ylabel('Loss (Train/Margin)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    fig.legend(lines, labels, loc='upper right')
    fig.suptitle('CapsNet Training Losses')
    fig.tight_layout()
    fig.savefig('capsule_training_loss.png')
    plt.close(fig)

    # Accuracy figure: left full-range, right zoom-in near top
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_full.plot(epochs, train_accuracies, label='Train Accuracy')
    ax_full.plot(epochs, test_accuracies, label='Test Accuracy')
    ax_full.set_title('Accuracy (full)')
    ax_full.set_xlabel('Epoch')
    ax_full.set_ylabel('Accuracy (%)')
    ax_full.set_ylim(0, 100)
    ax_full.grid(True, linestyle='--', alpha=0.6)
    ax_full.legend(loc='lower right')

    y_top = max([0.0] + train_accuracies + test_accuracies)
    y_bottom = max(0.0, y_top - 5.0)  # zoom to top 5% band
    ax_zoom.plot(epochs, train_accuracies, label='Train Accuracy')
    ax_zoom.plot(epochs, test_accuracies, label='Test Accuracy')
    ax_zoom.set_title('Accuracy (zoomed)')
    ax_zoom.set_xlabel('Epoch')
    ax_zoom.set_ylabel('Accuracy (%)')
    ax_zoom.set_ylim(y_bottom, 100)
    ax_zoom.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('CapsNet Training Accuracies')
    fig.tight_layout()
    fig.savefig('capsule_training_accuracy.png')
    plt.close(fig)

def _subplot_grid(axes, nrows, ncols):
    # normalize axes to 2D array
    if nrows == 1 and ncols == 1:
        return [[axes]]
    if nrows == 1:
        return [list(axes)]
    if ncols == 1:
        return [[ax] for ax in axes]
    return [list(row) for row in axes]

def collect_error_cases(model, decoder, loader, device, max_cases=64):
    model.eval()
    decoder.eval()
    error_images = []
    error_trues = []
    error_preds = []
    error_recons = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            caps_output = model(data)
            v_lengths = (caps_output ** 2).sum(dim=-1).sqrt()
            pred = v_lengths.argmax(dim=1)
            mask = pred.ne(target)
            if mask.any():
                idxs = torch.nonzero(mask).squeeze(1)
                for idx in idxs:
                    img = data[idx].detach().cpu().squeeze(0)  # [1,28,28] -> [28,28]
                    y_pred = pred[idx].unsqueeze(0)
                    recon = decoder(caps_output[idx:idx+1], y_pred)
                    recon = recon.view(28, 28).detach().cpu()
                    error_images.append(img)
                    error_trues.append(int(target[idx].item()))
                    error_preds.append(int(pred[idx].item()))
                    error_recons.append(recon)
                    if len(error_images) >= max_cases:
                        break
            if len(error_images) >= max_cases:
                break
    return error_images, error_trues, error_preds, error_recons

def save_error_grid(images, trues, preds, filename, ncols=8):
    import math
    n = len(images)
    if n == 0:
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, 'No misclassified samples found', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.8))
    grid_axes = _subplot_grid(axes, nrows, ncols)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = grid_axes[r][c]
            ax.axis('off')
            if idx < n:
                ax.imshow(images[idx], cmap='gray')
                ax.set_title(f'T={trues[idx]} P={preds[idx]}', fontsize=8)
            idx += 1
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- 1. SQUASH 激活函数 ---
# 这是 Capsule 论文中提出的非线性激活函数
# 它将向量的模长压缩到 0-1 之间，同时保持其方向
def squash(s, dim=-1):
    """
    Squash 激活函数：v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
    :param s: 输入张量 (向量)
    :param dim: 沿着哪个维度计算模长
    """
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * s / torch.sqrt(squared_norm + 1e-8)

# --- 2. DigitCaps 模块 (动态路由) ---
class DigitCaps(nn.Module):
    """
    高层胶囊（DigitCaps），实现动态路由算法
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, num_routing):
        """
        :param in_num_caps: 输入胶囊的数量 (例如 1152)
        :param in_dim_caps: 输入胶囊的维度 (例如 8)
        :param out_num_caps: 输出胶囊的数量 (例如 10，代表 10 个数字)
        :param out_dim_caps: 输出胶囊的维度 (例如 16)
        :param num_routing: 动态路由的迭代次数 (例如 3) [cite: 146]
        """
        super(DigitCaps, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.num_routing = num_routing
        
        # 变换矩阵 W (shape: 1, in_num, out_num, out_dim, in_dim)
        # 我们使用 nn.Parameter 使其可训练
        self.W = nn.Parameter(torch.empty(1, in_num_caps, out_num_caps, out_dim_caps, in_dim_caps))
        nn.init.normal_(self.W, mean=0.0, std=0.01)

    def forward(self, u):
        """
        :param u: PrimaryCaps 的输出 (shape: [batch_size, in_num_caps, in_dim_caps])
        :return: DigitCaps 的输出 (shape: [batch_size, out_num_caps, out_dim_caps])
        """
        batch_size = u.size(0)
        
        # 扩展 u 的维度以进行矩阵乘法
        # u_expanded shape: [batch_size, 1152, 1, 8, 1]
        u_expanded = u.unsqueeze(2).unsqueeze(4)

        # 计算"预测向量" u_hat
        # W shape: [1, 1152, 10, 16, 8]
        # u_hat shape: [batch_size, 1152, 10, 16, 1] -> [batch_size, 1152, 10, 16]
        u_hat = torch.matmul(self.W, u_expanded).squeeze(4)

        # 动态路由 [cite: 71]
        # b (logits) 初始化为 0，shape: [batch_size, 1152, 10]
        b = torch.zeros(batch_size, self.in_num_caps, self.out_num_caps, device=u.device) # [cite: 71, 125]

        for i in range(self.num_routing):
            # c (coupling coefficients) 是 b 的 softmax
            # c shape: [batch_size, 1152, 10]
            c = F.softmax(b, dim=2) # [cite: 63, 71]
            
            # s (weighted sum) 是 c 和 u_hat 的加权和
            # s shape: [batch_size, 1, 10, 16]
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1, keepdim=True) # [cite: 54, 71]
            
            # v (squashed output)
            # v shape: [batch_size, 1, 10, 16]
            v = squash(s, dim=-1) # [cite: 50, 71]

            # 更新 b (agreement)
            if i < self.num_routing - 1:
                # v_tiled shape: [batch_size, 1152, 10, 16]
                v_tiled = v.repeat(1, self.in_num_caps, 1, 1)
                
                # agreement (dot product) shape: [batch_size, 1152, 10]
                agreement = (u_hat * v_tiled).sum(dim=-1) # [cite: 68]
                
                b = b + agreement # [cite: 71]

        # 返回时去掉多余的维度
        # v shape: [batch_size, 10, 16]
        return v.squeeze(1)

# --- 3. 重建器 (Decoder) ---
# 这是一个简单的全连接网络，用于重建损失的正则化 [cite: 128]
class Decoder(nn.Module):
    def __init__(self, caps_dim=16, num_caps=10, img_size=28):
        super(Decoder, self).__init__()
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.img_pixels = img_size * img_size
        
        # 论文中描述的解码器架构 [cite: 132, 92]
        self.decoder = nn.Sequential(
            nn.Linear(num_caps * caps_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.img_pixels),
            nn.Sigmoid()  # 确保像素值在 0-1 之间 [cite: 92, 114]
        )
        self._init_weights()
    
    def forward(self, x, y):
        """
        :param x: DigitCaps 的输出 (shape: [batch_size, 10, 16])
        :param y: 真实标签 (shape: [batch_size])
        :return: 重建的图像 (shape: [batch_size, 784])
        """
        batch_size = x.size(0)
        
        # 1. 蒙版 (Masking) [cite: 130]
        # 我们只使用“正确”的胶囊来进行重建 [cite: 130]
        
        # 创建 one-hot 标签
        # y_one_hot shape: [batch_size, 10]
        y_one_hot = F.one_hot(y, num_classes=self.num_caps).float()
        
        # y_mask shape: [batch_size, 10, 1]
        y_mask = y_one_hot.unsqueeze(2)
        
        # (x * y_mask) shape: [batch_size, 10, 16]
        # 只保留了正确类别的胶囊向量，其余都为 0
        masked_caps = x * y_mask
        
        # 2. 展平并重建
        # decoder_input shape: [batch_size, 160]
        decoder_input = masked_caps.view(batch_size, -1)
        
        # reconstructed_img shape: [batch_size, 784]
        reconstructed_img = self.decoder(decoder_input)
        return reconstructed_img

    def _init_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# --- 4. 完整的 CapsNet 模型 ---
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        
        # 论文中的架构 [cite: 81]
        # Conv1: 256 个 9x9 卷积核, stride 1, ReLU [cite: 82]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        
        # PrimaryCaps: 32 个 8D 胶囊的卷积层 [cite: 86]
        # 论文描述为 32 channels of 8D capsules, 9x9 kernel, stride 2 [cite: 86]
        # 这等价于一个 32*8 = 256 个输出通道的 Conv2d
        self.primary_caps = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2)
        
        # MNIST 28x28 输入:
        # (28-9+1) = 20 -> conv1 out: (256, 20, 20) [cite: 92]
        # (20-9+0)/2 + 1 = 6 (约) -> primary_caps out: (32*8, 6, 6) [cite: 92]
        
        # 总的 PrimaryCaps 数量 = 32 * 6 * 6 = 1152 [cite: 120]
        self.digit_caps = DigitCaps(
            in_num_caps=32 * 6 * 6, # [cite: 120]
            in_dim_caps=8,           # [cite: 86]
            out_num_caps=10,         # [cite: 122]
            out_dim_caps=16,         # [cite: 122]
            num_routing=6           # [cite: 146]
        )

        self._init_weights()

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        
        # Conv1
        # out shape: [batch_size, 256, 20, 20]
        x = F.relu(self.conv1(x), inplace=True) # [cite: 82]
        
        # PrimaryCaps
        # out shape: [batch_size, 256 (32*8), 6, 6]
        primary_caps_out = self.primary_caps(x) # [cite: 86]
        
        # 变形以匹配 PrimaryCaps 的逻辑
        batch_size = primary_caps_out.size(0)
        
        # out shape: [batch_size, 1152 (32*6*6), 8]
        u = primary_caps_out.view(batch_size, 32 * 6 * 6, 8) # [cite: 120]
        
        # 应用 squash 激活 [cite: 121]
        u = squash(u)
        
        # DigitCaps
        # out shape: [batch_size, 10, 16]
        caps_output = self.digit_caps(u) # [cite: 123]
        
        return caps_output

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.primary_caps.weight, nonlinearity='relu')
        if self.primary_caps.bias is not None:
            nn.init.zeros_(self.primary_caps.bias)

# --- 5. Capsule 损失函数 ---
class CapsuleLoss(nn.Module):
    def __init__(self, decoder, m_plus=0.9, m_minus=0.1, lambda_val=0.5, recon_alpha=0.0005):
        """
        :param decoder: 传入的解码器
        :param m_plus: 论文中的 m+ [cite: 78]
        :param m_minus: 论文中的 m- [cite: 78]
        :param lambda_val: 论文中的 lambda [cite: 79]
        :param recon_alpha: 重建损失的缩放系数 [cite: 133]
        """
        super(CapsuleLoss, self).__init__()
        self.decoder = decoder
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.recon_alpha = recon_alpha
        self.reconstruction_loss_fn = nn.MSELoss(reduction='sum') # 论文中使用的是 Sum of squared differences [cite: 132]

    def forward(self, x_caps, y_true, x_original):
        """
        :param x_caps: CapsNet 的输出 (shape: [batch_size, 10, 16])
        :param y_true: 真实标签 (shape: [batch_size])
        :param x_original: 原始输入图像 (shape: [batch_size, 1, 28, 28])
        :return: 总损失
        """
        batch_size = x_caps.size(0)
        
        # --- 边距损失 (Margin Loss) --- [cite: 75]
        
        # v_lengths (模长) shape: [batch_size, 10]
        v_lengths = torch.sqrt((x_caps ** 2).sum(dim=-1) + 1e-8)
        
        # one-hot 标签
        y_one_hot = F.one_hot(y_true, num_classes=10).float()
        
        # L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2 [cite: 76]
        
        # 正确类别的损失 (T_k = 1)
        loss_plus = (y_one_hot * F.relu(self.m_plus - v_lengths) ** 2).sum(dim=1) # [cite: 76]
        
        # 错误类别的损失 (T_k = 0)
        loss_minus = (self.lambda_val * (1 - y_one_hot) * F.relu(v_lengths - self.m_minus) ** 2).sum(dim=1) # [cite: 76]
        
        # 边距损失（在 batch 上取平均）
        margin_loss = (loss_plus + loss_minus).mean() # [cite: 79]
        
        # --- 重建损失 (Reconstruction Loss) --- [cite: 128]
        
        # 1. 获取重建图像
        reconstructed_img = self.decoder(x_caps, y_true) # [cite: 131]
        
        # 2. 展平原始图像
        x_original_flat = x_original.view(batch_size, -1)
        
        # 3. 计算 MSE (论文中用 SSE) [cite: 132]
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed_img, x_original_flat)
        
        # 总损失 = 边距损失 + alpha * 重建损失 [cite: 133]
        # 注意：重建损失要除以 batch_size 来取平均
        total_loss = margin_loss + self.recon_alpha * reconstruction_loss / batch_size
        
        return total_loss, margin_loss, reconstruction_loss / batch_size

# --- 6. 训练与测试 ---

def train(model, decoder, loss_fn, optimizer, train_loader, device):
    model.train()
    decoder.train()
    total_loss = 0
    total_margin_loss = 0
    total_recon_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        caps_output = model(data)
        
        # 计算损失
        loss, margin_loss, recon_loss = loss_fn(caps_output, target, data)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_margin_loss += margin_loss.item()
        total_recon_loss += recon_loss.item()
        
        # 计算准确率
        v_lengths = (caps_output ** 2).sum(dim=-1).sqrt()
        pred = v_lengths.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    avg_margin_loss = total_margin_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    print(f'Train: Avg. Loss: {avg_loss:.4f} | Margin Loss: {avg_margin_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | Accuracy: {accuracy:.2f}%')
    return avg_loss, avg_margin_loss, avg_recon_loss, accuracy

def test(model, decoder, loss_fn, test_loader, device):
    model.eval()
    decoder.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            caps_output = model(data)
            
            # 计算损失
            loss, _, _ = loss_fn(caps_output, target, data)
            test_loss += loss.item()
            
            # 计算准确率
            # 预测结果是模长最长的那个胶囊 [cite: 73, 93]
            v_lengths = (caps_output ** 2).sum(dim=-1).sqrt()
            pred = v_lengths.argmax(dim=1) # (shape: [batch_size])
            
            correct += pred.eq(target).sum().item()
            
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest Set: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

class LocalMNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = int(img_name.split('_')[0])
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # --- 超参数和设置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    BATCH_SIZE = 128
    EPOCHS = 200 # 论文中没讲训练了多少次，就实验结果来看差不多100次就收敛了，那放100次试试。
    LR = 0.001 # 论文中使用 Adam，lr=0.001 [cite: 126]

    # --- 数据加载 ---
    
    # 论文中提到的数据增强： 
    # "shifted by up to 2 pixels in each direction with zero padding."
    # 我们通过 2 像素填充（0值）+ 28x28 随机裁剪来实现
    train_transform = transforms.Compose([
        transforms.Pad(2, fill=0),
        transforms.RandomCrop(28),
        transforms.ToTensor()
        # 不使用 transforms.Normalize()，因为重建器使用 Sigmoid
    ])
    
    # 测试集不需要增强，保持原始状态
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = LocalMNIST('1-Digit-TrainSet/TrainingSet', transform=train_transform)
    test_dataset = LocalMNIST('1-Digit-TestSet/TestSet', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 初始化模型、损失和优化器 ---
    model = CapsNet().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    
    capsule_loss_fn = CapsuleLoss(decoder, recon_alpha=0.0005).to(DEVICE) # [cite: 133]
    
    # 优化器需要同时优化模型和解码器的参数
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR) # [cite: 126]
    
    # 学习率调度器（论文中提到使用，但默认参数） [cite: 126]
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96) # 示例
    
    # --- 训练循环 ---
    best_acc = 0
    best_epoch = 0
    best_model_state = None
    best_decoder_state = None
    train_losses = []
    train_margin_losses = []
    train_recon_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        avg_loss, avg_margin_loss, avg_recon_loss, train_acc = train(model, decoder, capsule_loss_fn, optimizer, train_loader, DEVICE)
        train_losses.append(avg_loss)
        train_margin_losses.append(avg_margin_loss)
        train_recon_losses.append(avg_recon_loss)
        train_accuracies.append(train_acc)
        accuracy = test(model, decoder, capsule_loss_fn, test_loader, DEVICE)
        test_accuracies.append(accuracy)
        # 保存当前曲线（即使中途停止也能得到正确纵轴与最新数据）
        save_training_curves(train_losses, train_margin_losses, train_recon_losses, train_accuracies, test_accuracies)
        
        # 简单的学习率衰减
        # scheduler.step()
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            # 保存当前最佳模型/解码器的权重（用于后续错误样本可视化）
            best_model_state = deepcopy(model.state_dict())
            best_decoder_state = deepcopy(decoder.state_dict())
            # 你可以在这里保存模型
            # torch.save(model.state_dict(), 'capsnet_model.pth')
            # torch.save(decoder.state_dict(), 'capsnet_decoder.pth')
            
    print(f"训练完成. 最佳测试准确率: {best_acc:.2f}% | 出现在第 {best_epoch} 轮")

    # 结束后再保存一次，确保最终曲线完整
    save_training_curves(train_losses, train_margin_losses, train_recon_losses, train_accuracies, test_accuracies)

    # --- 错误样本可视化 ---
    print('收集并可视化错误样本...')
    # 使用“最佳测试准确率”对应的权重进行推理
    if best_model_state is not None and best_decoder_state is not None:
        model.load_state_dict(best_model_state)
        decoder.load_state_dict(best_decoder_state)
    err_imgs, err_trues, err_preds, err_recons = collect_error_cases(model, decoder, test_loader, DEVICE, max_cases=64)
    save_error_grid(err_imgs, err_trues, err_preds, 'capsule_error_cases.png', ncols=8)
    save_error_grid(err_recons, err_trues, err_preds, 'capsule_error_cases_recon.png', ncols=8)
    print(f"错误样本数量: {len(err_imgs)} | 已保存: capsule_error_cases.png, capsule_error_cases_recon.png")

if __name__ == "__main__":
    main()
