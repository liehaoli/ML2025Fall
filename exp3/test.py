import torch
from torch.utils.data import Dataset, DataLoader
from metric import calc_cc_score, KLD
import cv2
import numpy as np
import os
from tqdm import tqdm
from model import ResNet18Saliency
from dataset import SaliencyDataset
from collections import defaultdict

class SaliencyTestDataset(SaliencyDataset):
    def __init__(self, root_dir, target_size=320, is_train=False):
        super().__init__(root_dir, img_size=(target_size, target_size), is_train=is_train)
        self.target_size = target_size

    def __getitem__(self, idx):
        # 读取原图并记录尺寸
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]

        # 1. 保持长宽比缩放 (Long-side resize)
        # 将长边缩放到 target_size，短边按比例缩放
        scale = self.target_size / max(ori_h, ori_w)
        new_h, new_w = int(ori_h * scale), int(ori_w * scale)
        img_resized = cv2.resize(img_ori, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. Padding 到 32 的倍数
        # 计算需要的尺寸（向上取整到32倍数）
        pad_h_target = (new_h + 31) // 32 * 32
        pad_w_target = (new_w + 31) // 32 * 32
        
        pad_h = pad_h_target - new_h
        pad_w = pad_w_target - new_w
        
        # 使用反射填充避免边缘突变
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        img = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0

        # 读取掩码用于计算指标
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)
        
        # 返回:
        # 1. 处理后的tensor
        # 2. (resize后的高, resize后的宽) -> 用于去除padding
        # 3. (原图高, 原图宽) -> 用于还原尺寸
        # 4. 原始掩码
        # 5. 原始图像
        # 6. 路径
        return img, (new_h, new_w), (ori_h, ori_w), mask_ori, img_ori, img_path

def overlay_saliency_on_image(original_img, saliency_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # 将显著性图归一化到0-255
    saliency_normalized = ((saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)) * 255
    saliency_normalized = saliency_normalized.astype(np.uint8)

    # 应用颜色映射
    saliency_colored = cv2.applyColorMap(saliency_normalized, colormap)

    # 图像叠加
    overlayed_img = cv2.addWeighted(saliency_colored, alpha, original_img, 1 - alpha, 0)

    return overlayed_img, saliency_colored

@torch.no_grad()
def test_and_evaluate(model, test_dir, save_dir="saliency_results", target_size=320):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # 使用自定义测试Dataset
    test_dataset = SaliencyTestDataset(test_dir, target_size=target_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    metrics_by_category = defaultdict(lambda: {"cc": [], "kl": []})
    all_cc = []
    all_kl = []

    pbar = tqdm(test_loader, desc="测试与评估")
    for idx, (img, (resized_h, resized_w), (ori_h, ori_w), mask_ori, img_ori, img_path_tuple) in enumerate(pbar):
        img_path = img_path_tuple[0]
        category = os.path.basename(os.path.dirname(img_path))
        cate_save_dir = os.path.join(save_dir, category)
        os.makedirs(cate_save_dir, exist_ok=True)

        # 模型预测
        img = img.to(device)
        pred_map = model(img) # [1, 1, H_pad, W_pad]
        pred_map = pred_map.squeeze().cpu().numpy() # [H_pad, W_pad]

        # 1. 去除Padding：截取有效区域
        rh, rw = resized_h.item(), resized_w.item()
        saliency_no_pad = pred_map[:rh, :rw]

        # 2. 还原回原图尺寸
        oh, ow = ori_h.item(), ori_w.item()
        saliency_pred_ori = cv2.resize(saliency_no_pad, (ow, oh))
        
        mask_ori = mask_ori.squeeze().numpy()
        
        # 确保 mask 和 pred 尺寸一致
        if mask_ori.shape != saliency_pred_ori.shape:
             saliency_pred_ori = cv2.resize(saliency_pred_ori, (mask_ori.shape[1], mask_ori.shape[0]))

        # 计算指标
        cc_score = calc_cc_score(mask_ori, saliency_pred_ori)
        kl_score = KLD(mask_ori, saliency_pred_ori)
        
        metrics_by_category[category]["cc"].append(cc_score)
        metrics_by_category[category]["kl"].append(kl_score)
        all_cc.append(cc_score)
        all_kl.append(kl_score)

        # 保存结果
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        saliency_pred_save = ((saliency_pred_ori - saliency_pred_ori.min()) / (saliency_pred_ori.max() - saliency_pred_ori.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(cate_save_dir, f"{img_name}.png"), saliency_pred_save)

        img_ori_np = img_ori.squeeze().numpy()
        img_ori_bgr = cv2.cvtColor(img_ori_np, cv2.COLOR_RGB2BGR)
        
        alpha = 0.3
        overlayed_img, _ = overlay_saliency_on_image(
            original_img=img_ori_bgr,
            saliency_map=saliency_pred_ori,
            alpha=alpha,
            colormap=cv2.COLORMAP_JET
        )

        cv2.imwrite(os.path.join(cate_save_dir, f"{img_name}_overlay.png"), overlayed_img)

    print("\n" + "="*50)
    print(f"{'Category':<20} | {'mCC':<10} | {'mKL':<10}")
    print("-" * 50)
    
    results_txt = []
    results_txt.append(f"{'Category':<20} | {'mCC':<10} | {'mKL':<10}")
    
    for cat, metrics in sorted(metrics_by_category.items()):
        avg_cat_cc = np.mean(metrics["cc"])
        avg_cat_kl = np.mean(metrics["kl"])
        print(f"{cat:<20} | {avg_cat_cc:.4f}     | {avg_cat_kl:.4f}")
        results_txt.append(f"{cat:<20} | {avg_cat_cc:.4f}     | {avg_cat_kl:.4f}")
        
    avg_cc = np.mean(all_cc)
    avg_kl = np.mean(all_kl)
    print("-" * 50)
    print(f"{'Overall':<20} | {avg_cc:.4f}     | {avg_kl:.4f}")
    results_txt.append("-" * 50)
    results_txt.append(f"{'Overall':<20} | {avg_cc:.4f}     | {avg_kl:.4f}")

    with open(os.path.join(save_dir, "metrics_detailed.txt"), "w") as f:
        f.write("\n".join(results_txt))

    return avg_cc, avg_kl

if __name__ == "__main__":
    TEST_DIR = "./data/3-Saliency-TestSet"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "resnet18_saliency_best.pth"
    
    # 使用与训练时相同或接近的尺寸进行测试，保证Scale一致性
    # 训练是 320x320，这里设置长边为 320，保持宽高比
    TARGET_SIZE = 320 

    model = ResNet18Saliency(pretrained=True).to(DEVICE)

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n加载最佳模型（Epoch {checkpoint['epoch']+1}）")
        test_and_evaluate(model, TEST_DIR, save_dir="saliency_results", target_size=TARGET_SIZE)
        print("测试完成！结果已保存至 saliency_results 目录")
    else:
        print(f"未找到模型文件 {SAVE_PATH}，请先运行 main.py 进行训练。")
