import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import ResNet18Saliency
from dataset import SaliencyDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="training")
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="validation")
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 配置参数
    TRAIN_DIR = "./data/3-Saliency-TrainSet"  # 训练集根目录
    TEST_DIR = "./data/3-Saliency-TestSet"  # 测试集根目录（需与训练集结构一致）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "resnet18_saliency_best.pth"  # 最佳模型保存路径


    IMG_SIZE = (320, 320)  # 图像尺寸 (增大尺寸以保留更多细节，需注意显存)
    BATCH_SIZE = 16  # 批次大小
    EPOCHS = 20  # 训练轮数 (增加轮数以更好拟合)
    LR = 1e-4  # 学习率 (适当降低初始学习率，配合余弦退火)
    criterion = nn.BCELoss() # 损失函数 (改为BCELoss更适合二分类/概率图)

    model = ResNet18Saliency(pretrained=True).to(DEVICE) # 初始化模型
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # 优化器 (使用AdamW通常效果更好)
    
    # 学习率衰减 (改为余弦退火，使学习率平滑下降)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


    # 初始化数据集
    train_dataset = SaliencyDataset(TRAIN_DIR, img_size=IMG_SIZE, is_train=True)
    val_dataset = SaliencyDataset(TEST_DIR, img_size=IMG_SIZE, is_train=False)  # 若有独立验证集可替换路径
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)




    # 训练主循环
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # scheduler step usually comes after epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, SAVE_PATH)
            print(f"保存最佳模型（验证损失：{best_val_loss:.4f}）")
