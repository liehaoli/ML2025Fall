import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # 递归获取所有图像路径
        self.img_paths = []
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
        for root, _, files in os.walk(os.path.join(root_dir, "Stimuli")):
            for file in files:
                if file.lower().endswith(img_extensions):
                    self.img_paths.append(os.path.join(root, file))

        # 匹配掩码路径
        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            mask_path = os.path.splitext(mask_path)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_path + ext
                if os.path.exists(candidate):
                    self.mask_paths.append(candidate)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到{img_path}对应的掩码文件")

        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]) if is_train else None

        print(f"成功加载{len(self.img_paths)}个样本")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取原图
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]

        # -----------------------------------------------------------------
        # 核心修改：保持长宽比 + Pad 到固定尺寸 (为了支持 batch_size > 1)
        # -----------------------------------------------------------------
        
        # 目标是填满这个正方形盒子
        target_h, target_w = self.img_size # (320, 320)

        # 计算缩放比例：以长边为准缩放到 320
        # 这与 test.py 的逻辑 "scale = self.target_size / max(ori_h, ori_w)" 是完全一致的
        scale = min(target_w / ori_w, target_h / ori_h)
        
        new_w = int(ori_w * scale)
        new_h = int(ori_h * scale)

        # Resize 图像
        img_resized = cv2.resize(img_ori, (new_w, new_h))
        
        # 2. Padding 计算
        # 注意：这里和 test.py 不同！
        # test.py 是 pad 到 "最近的 32 倍数" (导致尺寸不固定)
        # train 必须 pad 到 "target_size" (固定 320x320)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        
        # 为了让物体居中（可选），这里我们简单地将 Padding 放到右下侧
        # 这与 test.py 的逻辑一致 (0, pad_h, 0, pad_w)
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        # 转 Tensor
        img = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0

        # 3. 处理 Mask (必须完全同步)
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)
        
        # Resize
        mask_resized = cv2.resize(mask_ori, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Pad (注意 Mask 填充必须是 0，即 BORDER_CONSTANT)
        mask_padded = cv2.copyMakeBorder(mask_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        # 转 Tensor
        mask = torch.from_numpy(mask_padded).unsqueeze(0).float() / 255.0

        # 4. 数据增强
        if self.transform and self.is_train:
            # 设定种子保证 img 和 mask 做同样的增强
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        return img, mask, (ori_h, ori_w), mask_ori, img_ori