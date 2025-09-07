---
title: "BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting"
date: 2025-09-07 17:57:00 +0800
categories: [3D Vision, Research]
tags: [3DGS, HANDS, Multi-View, Reconstruction, 手部重建]
math: true
DOI：: https://arxiv.org/abs/2312.02137v1
开源代码: https://github.com/On-JungWoan/BIGS/tree/main
image:
  path: /assets/images/BIGS.png
  alt: BIGS 双手和物体重建
---

## 核心

---

仅给一段**单目视频**，在**双手与未知物体**复杂交互、严重遮挡的情况下，同时重建**两只手**与**物体**的3D形状/姿态与交互关系，并能进行**新视角/新姿态渲染**。这是比“一只手 + 物体”更困难的设定。

其实就是相比于HOLD把SDF换成了Gaussian并做了双手的联合优化，本身创新点不多

---

## 问题/背景：

---

1. 目前已有的方法过于依赖一些先验，一些方法被限制在10-20种特定的的物品，难以涵盖真实世界大量物品的场景
2. HOLD方法基于SDF-一种NERF方法。MANUS方法需要多个视角

---

## 方法

---

### **核心思路**

使用**3D Gaussian Splatting (3DGS)作为统一显式表示；手与物体先分别优化**得到可靠几何，再进行**交互一致性优化**校正手—物体的3D对齐；其中特别用**扩散模型的SDS损失**修补物体在遮挡处的“看不见”表面，而手部则依赖**MANO先验 + 单一手高斯共享**来稳健恢复。

### 1) 预处理

- **相机与初始几何**：用层次定位/SfM得到相机与物体点云与位姿；手部用 HaMeR（Transformer-based regressor）回归 MANO 的姿态 θ、形状 β、全局旋转 Φ/平移 Γ。这些**网格顶点**用作高斯**初始位置**。

### 2) 手的高斯（canonical → posed）

- 在**canonical空间只**建一套"右手"高斯 $G_H$，通过**TriplaneNet 特征**+ 三个 MLP（几何 $f_G^H$、外观 $f_A^H$、**LBS权重** $f_D^H$）预测每个高斯的中心偏移、旋转、尺度、颜色、不透明度与**到各关节的LBS权重**。
- 利用 MANO 的骨架变换和 LBS 将 canonical 高斯变换到**姿态空间**（Eq. (3)），再用视频帧的($Φ_H^t, Γ_H^t$)送到**图像坐标**。
- **关键：单手共享**。左手通过**x 轴翻转**共享同一套右手 canonical 高斯，从而**在有限视角下"汇聚"双手信息**、缓解遮挡。

> **LBS 是什么？（Linear Blend Skinning，线性混合蒙皮）**
> 
> - 本质：把一个点（这里是**手的高斯中心**）当作会被多根骨骼/关节同时影响的“蒙皮点”，用一组**权重**对各关节的刚体变换做线性加权，得到该点在**姿态空间**的位置。
> - 公式（BIGS 用到的形式）：对第 iii 个手部高斯在规范空间的中心 μic\mu_i^{c}μic，其姿态空间中心
>     
>     $\mu_i^{p}=\sum_{k=1}^{K} W_k(\mu_i^{c})\bigl(\Phi_k\,\mu_i^{c}+\Gamma_k\bigr),$
>     
>     其中 KKK 是手的关节数；,$\Phi_k,\Gamma_k$ 是第 k 个关节的旋转与平移；$W_k(\cdot)$ 是该点受第 k 个关节影响的权重（各权重通常非负且和为1）
>     
>     https://zhuanlan.zhihu.com/p/693202505
>     

### 3) 物体的高斯（canonical）

- 物体也在 canonical 空间建高斯 $G_O$，用**TriplaneNet + 两个 MLP**（几何 $f_G^O$、外观 f_A^O）预测参数；再用帧级($Φ_O^t, Γ_O^t$)送到图像坐标。

### 4) 两阶段优化

### 阶段A：**单主体优化**（手/物体分开）

- **总目标**（Eq. (4)）：L_image + I_hand  L_hand + I_obj  L_obj。
- **图像一致性损失 L_image**：对手/物体前景（SAM2 得到mask）进行 L1 + SSIM + VGG 感知损失，并含**颜色/尺度正则**与**mask-outside惩罚**（避免漂移高斯），见 Eq. (5)。
- **手部损失 L_hand**：
    - 时间平滑：θ^t、Γ_H^t的相邻帧差分；
    - **LBS权重正则**：把预测的 LBS 权重拉向由 MANO 网格**近邻顶点插值**得到的伪真值（F-范数），见 Eq. (6)。
- **物体损失 L_obj**（**SDS**）：
    - 先做**文本反演**，学到"**A photo of <token> <object>**"式的提示词y；
    - 用 **PiDi-ControlNet**（边界条件）给扩散模型加**几何边界约束**；
    - 随机虚拟相机绕 canonical 物体渲染前景图，再用**SDS梯度**推动物体高斯补全不可见面（Eq. (7)）。

### 阶段B：**交互主体优化**（对齐接触）

- 观察：遮挡与少视角会导致手/物体在3D里**错位**，影响接触。
- 做法：仅优化每帧**双手平移 $Γ_L^t, Γ_R^t$**，加入**接触正则 L_contact**——鼓励手与物体在空间上靠近（权重较小，λ=1.0），总目标 Eq. (9)。

### 5) 渲染与动画

- 一旦得到手/物体高斯，即可在**新手姿 / 新物体姿 / 新机位**下渲染视频与图像（图1、图3有示例）

---

## 实验

---

### 数据集

1. ARCTIC 
2. HO3Dv3 

### 指标

1. MPJPE
2. Chamfer distance (CDo)
3. F10
4. hand-relative Chamfer distance (CDl and CDr)
5. PSNR, SSIM, LPIPS

---

## 贡献/成果

---

- 提出**BIGS**：面向“双手 + 未知物体 + 单目视频”的3DGS重建管线；
- **两阶段优化**：先“单主体”（手/物体各自）再“交互主体”（对齐接触）；
- **物体端**引入**SDS**（文本反演 + PiDi-ControlNet 约束）以弥补不可见面；**手端**共享一套右手的**canonical高斯**并镜像到左手；
- 在 ARCTIC（双手）与 HO3Dv3（一手）上取得**SOTA**的手/物体重建、接触重建与渲染质量。

---