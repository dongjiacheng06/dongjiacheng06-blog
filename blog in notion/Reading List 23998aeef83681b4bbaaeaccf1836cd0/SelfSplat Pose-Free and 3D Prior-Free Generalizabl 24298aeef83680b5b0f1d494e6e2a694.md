# SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting

Status: To read
DOI：: https://arxiv.org/abs/2411.17190

## 问题/背景：

---

现有 **3D Gaussian Splatting (3D-GS)** 方法依赖**精确相机位姿**和**逐场景优化**，无法直接处理“野外”视频；而 NeRF-系 pose-free 工作又常需预训练模型或后期微调，且体渲渲染开销大。SelfSplat 旨在同时解决 **“无位姿、无 3D 先验、一次前向即可重建”** 的难题，兼顾速度与质量。

---

## 方法

---

![image.png](SelfSplat%20Pose-Free%20and%203D%20Prior-Free%20Generalizabl%2024298aeef83680b5b0f1d494e6e2a694/image.png)

### SelfSplat 方法详解（CVPR 2025）

> 任务设定输入为一组三张未标定帧 $(I_{c1}, I_t, I_{c2})$。网络须**一次前向**同时预测
> 
> 1. 两个上下文帧的 **像素对齐 Gaussians** $G_{c1},G_{c2}$；
> 2. 相对位姿 $T_{c1\!\to t},T_{c2\!\to t}$；
> 3. 连贯的深度图。

---

### 1 总体框架

| 模块 | 作用 | 关键实现 |
| --- | --- | --- |
| **多视 CNN + Swin 编码器** | 提取跨视图特征 $F^{\text{mv}}$ | ResNet 下采样 × 6 层 Swin，局部–跨窗注意力 |
| **单目 CroCo-ViT 编码器** | 补充纹理/弱视差场景信息 $F^{\text{mono}}$ | 共享权重 CroCo-v2，16×16 patch |
| **DPT 融合 & 密集预测** | 融合多/单视特征并输出 • 初始深度 $\tilde D_{k}$ • 高斯属性 $\tilde G_{k}$ | Dense-Prediction Transformer + 金字塔 CNN |
| **Matching-aware Pose Net** | 高精度相机位姿 | 2D U-Net + cross-attention，输入匹配特征 $F^{\text{ma}}$ 与射线嵌入 $E_{\text{int}}$ |
| **Pose-aware Depth Refine** | 跨视一致深度 $D_k$ | 轻量 U-Net，利用 Plücker 射线嵌入 $E_{\text{ext}}(T)$ |
| **Gaussian Decoder** | δ-offset→3D 反投影，合并到目标坐标系 | 公式 (4) |

---

### 2 像素对齐 3D Gaussians

密集预测模块输出

$\tilde G_k=\{(\delta x_j,\delta y_j,\alpha_j,\Sigma_j,c_j)\}_{j=1}^{HW},\quad
\tilde D_k$

将 $(\delta x_j,\delta y_j)$加到对应像素坐标，再用精化后深度 $D_k$ 反投影到 3D，生成每帧的 $G_k$。随后通过预测位姿把 $G_{c1},G_{c2}$ 变换到目标视角统一成 $G$。

---

### 3 匹配感知位姿网络

- **跨视匹配**：先用 MatchingNet 输出同分辨率特征 $F^{\text{ma}}_{c1},F^{\text{ma}}t,F^{\text{ma}}{c2}$。
- **射线嵌入**：每像素拼接 $E_{\text{int}}(K^{-1}p)$提供尺度信息。
- **PoseNet**：两分支 U-Net + cross-attn，同步预测 $T_{c1\!\to t}$ 与 $T_{c2\!\to t}$。

> 效果：去掉匹配模块，平移误差↑1.5°，PSNR↓0.4 dB。
> 

---

### 4 基于位姿的深度精化

初始深度常在多视间不一致，导致高斯重叠。

- 将 $\tilde D_k$、原图 $I_k$、以及 Plücker 射线嵌入 $E_{\text{ext}}(T_{k\!\to t})$输入 Refinement U-Net；
- 输出残差 $\Delta D_k$，得到一致深度 $D_k=\tilde D_k+\Delta D_k$。

去掉精化模块，PSNR↓0.6 dB，平移误差↑1.1°。

---

### 5 自监督训练目标

1. **重投影光度损失**

$\mathcal L_{\text{proj}}
=\text{pe}(I_t, I_{c1\!\to t}){\text{SSIM+L1}}
+\text{pe}(I_t, I{c2\!\to t})$

1. **3DGS 渲染损失**  
$\mathcal L_{\text{ren}}
=\!\!\!\sum_{I_k\in\{I_{c1},I_{c2},I_t\}}\!\!\!\gamma_1 (1-\text{SSIM})+\gamma_2\|I_k-\hat I_k\|_2$
2. **总损失**  
$\mathcal{L}{\text{total}}=\lambda_1\mathcal{L}{\text{proj}}+\lambda_2\mathcal{L}_{\text{ren}}$

渲染损失对位姿的显式梯度  
$\partial \mathcal L_{\text{ren}}/\partial t,\partial R$ 直接约束外参，使纯图像重投影易陷入的尺度/漂移歧义得到缓解。 

> **消融**：去掉 $\mathcal L_{\text{ren}}$会导致旋转误差×8、平移误差×5 

---

### 6 融合策略与优势

显式 3D 表征 ↔ 自监督几何 的双向增益：

- 3DGS 渲染梯度强化位姿 / 深度收敛；
- 可靠位姿 / 深度保证 Gaussians 精确对齐，提高渲染质量

---

### 7 训练设置一览

- 完全**无需任何 3D 先验**；CroCo-v2 权重亦为纯自监督。
- 数据集：RE10K、ACID、DL3DV；50 k iterations 训练即可。
- 单前向即可在测试时获得 3D 场景与新视角渲染，且推理速度优于 NeRF-系方法。

---

### 8 小结 – 你可借鉴的要点

1. **双流特征**（跨视 Swin + 单目 ViT）比纯 cost-volume 更稳健，可直接替换到你打算加入 SWIN-Transformer 的 3DGS 方案。
2. **渲染损失反向监督位姿**，避免依赖外部 Flow/SfM。
3. **Plücker-ray + Cross-Attention** 的深度精化，能在弱纹理区域保持多视一致性。

如需代码落地建议或公式推导细节，随时告诉我！

---

## 实验

---

![image.png](SelfSplat%20Pose-Free%20and%203D%20Prior-Free%20Generalizabl%2024298aeef83680b5b0f1d494e6e2a694/image%201.png)

---

## 贡献/成果

---

- **提出 SelfSplat**：首个 *pose-free & 3D-prior-free* 的可泛化 3D-GS 框架；
- **自监督深度/位姿 × 显式 3D-GS** 的统一训练策略，实现几何-外参互相提升；
- **Matching-aware Pose Network** 引入跨视图注意力，提高位姿精度；
- **Pose-aware Depth Refinement** 通过 Plücker-ray embedding 消除多视深度不一致；
- 在 RealEstate10K、ACID、DL3DV 三大数据集上全面超越 SOTA，并展现强跨数据集泛化。

---