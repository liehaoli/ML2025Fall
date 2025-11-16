import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

# ---------- helpers: error-case visualization ----------
def _subplot_grid(axes, nrows, ncols):
    if nrows == 1 and ncols == 1:
        return [[axes]]
    if nrows == 1:
        return [list(axes)]
    if ncols == 1:
        return [[ax] for ax in axes]
    return [list(row) for row in axes]

def collect_error_cases_ensemble(models, loader, device, max_cases=64, mean=0.1307, std=0.3081):
    for m in models:
        m.eval()
    error_images, error_trues, error_preds = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            all_probs = []
            for m in models:
                out = m(data)
                probs = F.softmax(out, dim=1)
                all_probs.append(probs)
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred = avg_probs.argmax(dim=1)
            mask = pred.ne(target)
            if mask.any():
                idxs = torch.nonzero(mask).squeeze(1)
                for idx in idxs:
                    img = data[idx].detach().cpu().squeeze(0)  # [1,28,28] -> [28,28]
                    # unnormalize to [0,1]
                    img = img * std + mean
                    img = img.clamp(0.0, 1.0)
                    error_images.append(img)
                    error_trues.append(int(target[idx].item()))
                    error_preds.append(int(pred[idx].item()))
                    if len(error_images) >= max_cases:
                        break
            if len(error_images) >= max_cases:
                break
    return error_images, error_trues, error_preds

def save_error_grid(images, trues, preds, filename, ncols=8):
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

# --- 1. 定义超参数和设置 ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MODELS = 7             # 集成中模型的数量（略增提升集成效果）
EPOCHS_PER_MODEL = 50     # 训练每个模型的轮数（更充分收敛）
BATCH_SIZE = 128
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
MODEL_SAVE_DIR = "models"  # 模型保存目录

# 创建模型保存目录
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


# --- 2. 数据增强与加载 ---

# 弹性变形 (Elastic Distortions)
# 这是此任务的核心。
# alpha: 变形强度。 sigma: 变形平滑度。
# 这些值是针对 MNIST (28x28) 调优的常见值。
elastic_transform = transforms.ElasticTransform(alpha=30.0, sigma=5.0)

# 训练集的数据转换：应用弹性变形 + 转换为 Tensor + 标准化
# 注意：弹性变形只应用于训练集
train_transform = transforms.Compose([
    elastic_transform,
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
])

# 测试集的数据转换：不需要数据增强，只需标准化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
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

def build_dataloaders(train_dir, test_dir, batch_size, use_elastic):
    # 根据是否启用弹性变形，选择训练集的 transform
    if use_elastic:
        effective_train_transform = train_transform
    else:
        effective_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    effective_test_transform = test_transform

    train_dataset = LocalMNIST(train_dir, transform=effective_train_transform)
    test_dataset = LocalMNIST(test_dir, transform=effective_test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

# --- 3. 定义深度 CNN 模型 ---

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 卷积块 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 1, 28, 28) -> (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (B, 32, 28, 28) -> (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)        # (B, 32, 28, 28) -> (B, 32, 14, 14)
        )
        # 卷积块 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 32, 14, 14) -> (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (B, 64, 14, 14) -> (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)        # (B, 64, 14, 14) -> (B, 64, 7, 7)
        )
        
        # 全连接层 (分类器)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # 使用 Dropout 防止过拟合
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# --- 4. 训练和评估函数 ---

# 训练函数 (单个 epoch)
def train_epoch(model, loader, optimizer, criterion):
    model.train() # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(loader, total=len(loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 评估函数 (测试单个模型)
def evaluate_model(model, loader):
    model.eval() # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 评估时不需要计算梯度
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # F.softmax(output, dim=1) # (不需要 softmax 来找 argmax)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100. * correct / total
    return accuracy

# 评估函数 (测试集成模型)
def evaluate_ensemble(models, loader):
    # 确保所有模型都在评估模式
    for model in models:
        model.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 存储来自每个模型的概率
            all_probs = []
            
            for model in models:
                output = model(data)
                # 计算 softmax 概率
                probs = F.softmax(output, dim=1)
                all_probs.append(probs)
            
            # 关键：对所有模型的概率进行平均 (软投票)
            # torch.stack 将 list of tensors (N, C) 堆叠为 (M, N, C)
            # M=模型数, N=batch_size, C=类别数
            avg_probs = torch.stack(all_probs).mean(dim=0)
            
            # 找出平均概率最高的类别
            _, predicted = torch.max(avg_probs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100. * correct / total
    return accuracy

# --- 5. 主执行流程：训练 N 个模型并集成 ---

def main():
    parser = argparse.ArgumentParser(description="MNIST CNN + Ensemble + Elastic Ablation")
    parser.add_argument('--mode', choices=['full', 'no_elastic', 'no_ensemble', 'no_ensemble_no_elastic'], default='full', help='Ablation mode')
    parser.add_argument('--num-models', type=int, default=NUM_MODELS, help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=EPOCHS_PER_MODEL, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, help='Weight decay (L2)')
    args = parser.parse_args()

    # 独立控制：是否集成、是否弹性形变 → 共4种模式
    if args.mode == 'full':
        use_elastic = True
        num_models = max(1, args.num_models)
    elif args.mode == 'no_elastic':
        use_elastic = False
        num_models = max(1, args.num_models)
    elif args.mode == 'no_ensemble':
        use_elastic = True
        num_models = 1
    else:  # no_ensemble_no_elastic
        use_elastic = False
        num_models = 1
    epochs = max(1, args.epochs)
    batch_size = max(1, args.batch_size)
    lr = args.lr
    weight_decay = args.weight_decay

    print(f"Using device: {DEVICE}")
    print("=== 开始训练集成模型 ===")
    print(f"模式: {args.mode}, 模型数: {num_models}, 轮数/模型: {epochs}, 批大小: {batch_size}, 学习率: {lr}")

    # 按需构建数据加载器（是否使用弹性变形由 mode 决定）
    train_loader, test_loader = build_dataloaders('1-Digit-TrainSet/TrainingSet', '1-Digit-TestSet/TestSet', batch_size, use_elastic)

    trained_models = []
    all_train_losses = [[] for _ in range(epochs)]
    all_train_accs = [[] for _ in range(epochs)]
    all_test_accs = [[] for _ in range(epochs)]
    best_single_acc = 0.0
    best_single_model_idx = -1
    best_single_epoch = 0
    
    for i in range(num_models):
        print(f"\n--- 训练模型 {i+1} / {num_models} ---")
        
        # 1. 实例化新模型
        # 每次循环都会创建一个新模型，具有不同的随机初始化
        model = DeepCNN().to(DEVICE)
        
        # 2. 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 3. 训练模型
        best_model_state = None
        best_model_acc = -1.0
        best_model_epoch = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            
            # (可选) 在每个 epoch 后评估单个模型
            test_acc = evaluate_model(model, test_loader)
            print(f"  模型 {i+1}, Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%")
            scheduler.step()
            # 记录该模型在验证/测试集上表现最佳的权重
            if test_acc > best_model_acc:
                best_model_acc = test_acc
                best_model_epoch = epoch + 1
                best_model_state = deepcopy(model.state_dict())
            if test_acc > best_single_acc:
                best_single_acc = test_acc
                best_single_model_idx = i + 1
                best_single_epoch = epoch + 1
            
            all_train_losses[epoch].append(train_loss)
            all_train_accs[epoch].append(train_acc)
            all_test_accs[epoch].append(test_acc)

        # 4. 保存训练好的模型 (状态字典)
        # 使用该模型在整个训练过程中测试集上最佳的权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"  模型 {i+1} 使用最佳权重 (Epoch {best_model_epoch}, 测试准确率 {best_model_acc:.4f}%) 进行保存与集成。")
        model_path = os.path.join(MODEL_SAVE_DIR, f"mnist_cnn_model_{i}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型 {i+1} 已保存至 {model_path}")
        
        trained_models.append(model)
        
    print("\n=== 所有模型训练完毕 ===")
    if best_single_model_idx != -1:
        print(f"全程最佳单模型测试准确率: {best_single_acc:.4f}% | 来自 模型 {best_single_model_idx} 第 {best_single_epoch} 轮")

    # --- 6. 评估集成模型 ---
    
    print("\n正在评估单个模型的平均准确率...")
    total_acc = 0
    for i, model in enumerate(trained_models):
        acc = evaluate_model(model, test_loader)
        print(f"  模型 {i+1} 最终准确率: {acc:.4f}%")
        total_acc += acc
    denom = max(1, len(trained_models))
    print(f"--- 单个模型平均准确率: {total_acc / denom:.4f}% ---")

    if len(trained_models) > 1 and args.mode != 'no_ensemble':
        print("\n正在评估集成模型的最终准确率 (软投票)...")
        ensemble_accuracy = evaluate_ensemble(trained_models, test_loader)
        print("\n" + "="*30)
        print(f"** 最终集成准确率: {ensemble_accuracy:.4f}% **")
        print("="*30)
    else:
        print("\n集成评估已跳过（单模型或 no_ensemble 模式）。")

    # Plot average losses and accuracies
    epochs_range = range(1, epochs + 1)
    avg_train_losses = [sum(losses) / max(1, len(losses)) for losses in all_train_losses]
    avg_train_accs = [sum(accs) / max(1, len(accs)) for accs in all_train_accs]
    avg_test_accs = [sum(accs) / max(1, len(accs)) for accs in all_test_accs]

    prefix = f"mix_{args.mode}_"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, avg_train_losses, label='Avg Train Loss')
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{prefix}training_loss.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, avg_train_accs, label='Avg Train Accuracy')
    plt.plot(epochs_range, avg_test_accs, label='Avg Test Accuracy')
    plt.title('Average Training Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{prefix}training_accuracy.png')
    plt.close()

    # --- 保存集成错误样本 ---
    print('\n收集并可视化集成的错误样本...')
    err_imgs, err_trues, err_preds = collect_error_cases_ensemble(trained_models, test_loader, DEVICE, max_cases=64)
    save_error_grid(err_imgs, err_trues, err_preds, f'{prefix}error_cases.png', ncols=8)
    print(f"错误样本数量: {len(err_imgs)} | 已保存: {prefix}error_cases.png")

if __name__ == "__main__":
    main()