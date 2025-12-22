## 基于Difussion的图像生成
在课程提供的初始代码的基础上，进行了三个方面的优化：
1. 修复了初始代码的一些逻辑问题
2. 更改模型，从简单的Unet转向和DDPM论文里的实现对齐(https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)）
3. 调参优化，引入了余弦退火学习率调度等等。

实验结果对比在record.md中
