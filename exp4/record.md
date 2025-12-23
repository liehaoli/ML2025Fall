我们对代码进行了四次重要的迭代改进，以下是核心优化总结：

1. **方法论重构：引入验证集 (Validation & Evaluation)**
   - **问题**：原代码仅监控 Training Loss，无法感知过拟合，属于“盲飞”状态。
   - **改进**：引入 CIFAR-10 测试集作为验证集，实现 `evaluate` 函数。现在通过 **Validation Loss** 作为评估和模型保存的唯一标准，确保模型泛化能力。

2. **工程健壮性：安全的模型保存策略 (Safe Checkpointing)**
   - **问题**：断点续训时会重置最佳 Loss 记录，导致优秀的旧模型被平庸的新模型覆盖。
   - **改进**：启动训练时自动读取硬盘上现存的最佳 Loss 值，实施**“仅当更优时覆盖”**策略，解决了断点续训的风险。

3. **架构大升级：完整版 DDPM U-Net (SOTA Architecture)**
   - **问题**：原 `SimpleUNet` 结构过于简陋（无 ResBlock，无 Attention，通道数少），性能上限比较低。
   - **改进**：改进了 `utils/unet.py`，复现 **DDPM 原论文 (Ho et al.)** 的核心架构：
     - 引入 **ResNetBlock** (GroupNorm + SiLU + Dropout)。
     - 在低分辨率层加入 **Multi-Head Self-Attention**。
     - 修正了 Time Embedding 的注入逻辑。

4. **训练策略优化 (Training Dynamics)**
   - **优化器**：从 `Adam` 升级为 **`AdamW`**，增加L2正则提升泛化性。
   - **调度器**：引入 **`CosineAnnealingLR`** (余弦退火)，让 Loss 在训练后期能精细收敛。
   - **稳定性**：加入 **Gradient Clipping (梯度裁剪)**，缓解由深层网络导致的梯度爆炸问题。

---

### 实验结果对比 (初步)

> **注意**：以下 FID 分数基于小样本 (Batch=64) 快速估算，数值偏高，主要用于横向对比趋势。

*   **基准模型 (Baseline)**:
    *   FID Score: **224.88**
    *   Inception Score: **2.67**

*   **改进模型 (Ours - Intermediate)**:
    *   FID Score: **198.4589** 
    *   Inception Score: **3.1853 ± 0.9792**



