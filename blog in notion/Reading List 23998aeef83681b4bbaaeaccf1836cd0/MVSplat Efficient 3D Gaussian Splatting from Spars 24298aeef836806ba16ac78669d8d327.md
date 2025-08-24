# MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

Status: Done
DOI：: https://arxiv.org/pdf/2403.14627

## 问题/背景：

---

传统的从图片生成3D场景的方法（如NeRF或3DGS）通常需要大量的输入图片（例如上百张）和漫长的“逐场景优化”过程，这在实际应用中非常不便 。近年来出现了一些“前馈”(feed-forward) 模型，它们通过在大型数据集上预训练，可以快速地对新场景进行3D重建，无需再次优化 。

然而，即便是最新的前馈模型（如论文中反复对比的

`pixelSplat`），在仅有两三张输入图片的情况下，也很难准确地重建出场景的3D几何结构 。

`pixelSplat` 直接从图像特征中推断深度的概率分布，这种方式较为模糊，容易产生有噪点和悬浮物的3D模型 。

---

## 方法

---

![image.png](MVSplat%20Efficient%203D%20Gaussian%20Splatting%20from%20Spars%2024298aeef836806ba16ac78669d8d327/image.png)

MVSplat 的核心目标是学习一个前馈网络fθ，该网络接收 K 个稀疏视角的图像 {Ii} 及其对应的相机内外参矩阵 {Pi} 作为输入，然后直接输出一组完整的3D高斯基元参数 。这些参数包括中心位置μj、不透明度 αj、协方差 Σj 和以球谐函数表示的颜色 cj 。

整个方法可以分为两个主要部分：**多视角深度估计** 和 **高斯参数预测**。

1. 多视角深度估计 (Multi-View Depth Estimation)

这是 MVSplat 方法的基石，目的是为了精准地预测 3D 高斯基元的中心位置μj 。该深度模型完全基于 2D 卷积和注意力机制，避免了计算开销大的 3D 卷积，因此非常高效 。它包含以下几个步骤：

- **多视角特征提取 (Multi-view feature extraction)**
    - 首先，使用一个浅层的类 ResNet 的 CNN 网络对每张输入图像进行特征提取，得到被 4 倍下采样的特征图 。
    - 然后，将这些特征图送入一个带有自注意力和交叉注意力层的多视角 Transformer 中 。这里采用的是 Swin Transformer 的局部窗口注意力机制以提高效率 。
    - 交叉注意力机制使得每个视图都能与其他所有视图交换信息，从而得到富含跨视图信息的特征{Fi} 。
    - 
    
    ![image.png](MVSplat%20Efficient%203D%20Gaussian%20Splatting%20from%20Spars%2024298aeef836806ba16ac78669d8d327/image%201.png)
    
- **代价体构建 (Cost-volume-construction)**
    - 这是方法的核心，通过“平面扫描（plane-sweep）”的方式来构建代价体，以编码不同深度候选值的跨视图特征匹配信息 。
    - 模型在预设的最近和最远深度范围之间，在逆深度域中均匀采样 D 个深度候选值{dm} 。
    - 对于参考视图 i，模型会将其他视图 j 的特征Fj 根据每个深度候选值 dm “变换”或“扭曲”到视图 i 的视角下，得到 D 个扭曲后的特Fdmj→i
    - 接着，计算参考视图 i 的原始特征Fi 和每个扭曲后的特征 Fdmj→i 之间的点积相似度（相关性），得到 D 个相关性图 Cdmi 。
    - 将所有 D 个相关性图堆叠起来，就构成了视图 i 的代价体Ci∈R4H×4W×D 。当有超过两个输入视图时，会计算所有其他视图与参考视图 i 的相关性，然后按像素取平均值 。
- **代价体优化 (Cost volume refinement)**
    - 由于在纹理较少的区域，初始代价体可能存在歧义，因此模型使用一个轻量级的 2D U-Net 来对其进行优化 。
    - U-Net 的输入是 Transformer 特征和初始代价体Ci 的拼接 。它输出一个残差ΔCi，加到初始代价体上得到优化后的代价体 C~i 。
    - 为了在不同视图的代价体之间交换信息，U-Net 在其最低分辨率的层中加入了交叉视图注意力层 。
- **深度估计 (Depth estimation)**
    - 优化后的代价体C~i 会被一个基于 CNN 的上采样器恢复到全分辨率 C^i 。
    - 模型对全分辨率代价体C^i 在深度维度上应用 softmax 操作，将其归一化为概率分布 。
    - 最后，通过对所有深度候选值进行加权平均，计算出最终的深度图Vi 。
- **深度优化 (Depth refinement)**
    - 为了进一步提升性能，模型引入了一个额外的深度优化步骤 。
    - 一个非常轻量级的 2D U-Net 会接收多视角图像、特征和当前预测的深度图作为输入，输出一个残差深度图 。
    - 将这个残差深度加到当前深度上，得到最终的深度输出 。这个U-Net同样在低分辨率层加入了交叉视图注意力 。

### 2. 高斯参数预测 (Gaussian Parameters Prediction)

在得到高精度的深度图后，模型会并行地预测高斯基元的其他参数：

- **高斯中心 μ**: 直接将多视角深度图利用相机参数反投影到三维世界坐标系中，形成点云 。然后将所有视图的点云合并，这些合并后的点云就作为 3D 高斯基元的中心 。
- **不透明度 α**: 在深度估计的 softmax 操作后，可以得到每个像素的匹配置信度（即 softmax 输出的最大值） 。这个置信度的物理意义与不透明度很相似（高置信度意味着该点很可能在物体表面），因此模型用两个卷积层从匹配置信度中预测不透明度α 。
- **协方差 Σ 和颜色 c**: 这两个参数由另外两个卷积层预测得出，其输入是图像特征、优化后的代价体和原始多视角图像的拼接 。协方差矩阵由一个缩放矩阵和一个旋转矩阵构成，颜色 c 则由预测出的球谐函数系数计算得到 。

### 3. 训练损失 (Training Loss)

MVSplat 的整个模型是端到端训练的，仅使用渲染图像和真实目标图像之间的光度损失（photometric loss）作为监督信号，无需任何真实的几何（如深度图）监督 。训练损失是

l2 损失和 LPIPS 损失的线性组合，权重分别为 1 和 0.05 。

![image.png](MVSplat%20Efficient%203D%20Gaussian%20Splatting%20from%20Spars%2024298aeef836806ba16ac78669d8d327/image%202.png)

---

## 实验

---

---

## 贡献/成果

---

---

**在 MVSplat 里，cost volume（代价体）可以理解成给“每张参考图像”构建的一座 H × W × D 三维记分册：**

1. **它记录了跨视图特征在不同深度假设上的相似度**。
    - 先在近-远深度范围内均匀取 D 个候选深度平面；对每一平面，把其他视图的特征按相机矩阵重投影到参考视图上。
    - 对重投影后的特征与参考视图特征做点积相关，得到一张“相似度图”。
    - 这样就为参考图像里的每个像素留下了 D 个“匹配分数”；把这 D 层堆叠起来，就得到尺寸 H/4 × W/4 × D 的 cost volume（论文后续再上采到全分辨率）。
2. **它的作用是把“几张图片之间该像素真正处于哪一深度”的信息显式交给网络**。高相似度意味着这些视图在该深度上观测到同一表面，因此 cost volume 为后续深度回归提供强几何线索。
3. **后端处理方式**
    - MVSplat 把 cost volume 与经过Transformer得到的跨视图特征在通道维拼接，送入轻量 2D U-Net 做细化；再对深度维做 softmax、求期望，直接输出稠密深度图。
    - 如果 cost volume 被移除，几何质量和渲染指标都会大幅下降，论文的消融实验专门验证了这一点。

**一句话**：在这篇文章中，cost volume 就是把“*这个像素在不同深度平面上的跨视图一致性*”编码成一个三维张量，网络只需在里面寻找最低代价（最高相似度）的位置，就能迅速推断出准确深度，从而为 3D 高斯中心定位打下坚实基础。

### ① Multi-View Transformer 是不是作者首创？

不是。把 **Self-Attention + Cross-View Attention** 用在多视图立体里，最早可追溯到 **TransMVSNet**（ICCV’21→CVPR’22）——该工作提出“Feature-Matching Transformer”，同样用自注意力聚合图内上下文、再用跨视注意力聚合其它视图特征([arXiv](https://arxiv.org/abs/2111.14600?utm_source=chatgpt.com), [CVF开放获取](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_TransMVSNet_Global_Context-Aware_Multi-View_Stereo_Network_With_Transformers_CVPR_2022_paper.pdf?utm_source=chatgpt.com))。MVSplat 延续了这一思想，只是把核心块换成轻量 **Swin-Transformer 窗口注意力** 并减少层数，以减小显存并保持实时推理；因此 **“跨视 Transformer” 不是新算法，而是对现有方案的工程化精简**，真正的新点在后端一次前向生成 3D Gaussians。

---

### ② 除了纯光度（L2 + LPIPS）还能加哪些无监督损失？

| 额外损失 | 作用 | 常见做法 / 文献 |
| --- | --- | --- |
| **SSIM 结构相似度** | 减少曝光差带来的像素级抖动；与 L1/L2 并用 | CL-MVSNet 在 photometric 损失里同时加 SSIM 项([arXiv](https://arxiv.org/html/2503.08219v1?utm_source=chatgpt.com)) |
| **深度平滑 / TV 正则** | 约束相邻像素深度差，抑制噪声 | 无监督 MVS 与单目估深普遍使用 edge-aware smooth loss([Tejas Khot](https://tejaskhot.github.io/static/docs/unmvs.pdf?utm_source=chatgpt.com), [欧洲计算机视觉协会](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08045.pdf?utm_source=chatgpt.com)) |
| **梯度一致性** | 强调图像梯度在重投影后也要一致，改善纹理边缘对齐 | “First-order gradient consistency” 于 UNMVS-TIJCV’21 提出([Tejas Khot](https://tejaskhot.github.io/static/docs/unmvs.pdf?utm_source=chatgpt.com)) |
| **前向-后向几何一致** | 利用已估深度把源图反投影，再与参考视图重投影求差，过滤遮挡 | 多数无监督 MVS 框架（GeoMVSNet、DIV-Loss 等）含此项([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_GeoMVSNet_Learning_Multi-View_Stereo_With_Geometry_Perception_CVPR_2023_paper.pdf?utm_source=chatgpt.com), [欧洲计算机视觉协会](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08045.pdf?utm_source=chatgpt.com)) |
| **视图选择 / 像素置信度** | 学习权重或掩码，忽略遮挡与外极面 | 早期 MVSNet 族用 **pixel-wise confidence** 或 **focal/ambiguity loss** 来降低误匹配([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_TransMVSNet_Global_Context-Aware_Multi-View_Stereo_Network_With_Transformers_CVPR_2022_paper.pdf?utm_source=chatgpt.com)) |

> 在 MVSplat 作者的实验里，这些附加约束会提升一点边缘精度，但会降低整体运行速度；为了保持 端到端 0.1 s 推理，论文最终只保留了 L2 + 0.05·LPIPS 两项光度误差，并在消融中说明加 Smooth/SSIM 收益有限而耗时显著增加。
> 
- **训练损失**：只用 **L2 + 0.05·LPIPS** 对比渲染图与真图，就能端到端学出深度和全部高斯参数，省去人工深度标注。
- **Multi-View Transformer**：在每层加入 self + cross-view attention，用 Swin 窗口高效交换视图信息，奠定后续几何一致基础。

### 一句话总结

---

> 为什么能只用光度误差？
> 
> - 通过上游 **cost volume + Transformer**，网络已把“对齐几何”线索嵌进特征；
> - 3D GS 渲染可直接微分到深度和高斯参数；
> - 因此只要让渲染结果在像素和感知两层与真图一致，网络就能同时收敛几何和外观，无需任何 GT 深度或法线。

| 项 | 公式/实现 | 目的 |
| --- | --- | --- |
| **像素级 L2（论文记 ℓ2）** | 对渲染图 Ĩ 与目标 RGB 图 I 逐像素求平方差并取平均；权重 **1.0** | 约束整体亮度、颜色一致 |
| **LPIPS 感知损失** | 用预训练 VGG16 等 backbone，在多层特征空间计算欧氏距离；论文权重 **0.05** | 捕捉纹理/结构相似度，弥补 L2 对高频不敏感的缺陷 |
| **总损失** | **L = L2 + 0.05 × LPIPS**(论文未使用 L1；若口头表述为 “L1”，实为经典像素重建项，此处实现是 L2) | 仅靠可微渲染图与真实图对比即可反向传播，**完全不需要深度 GT**；与 pixelSplat 相比省去额外标签 |

### 2. 纯光度损失（L2 + LPIPS）的组成与原因

---

Self-attention保证每张图自身的纹理连贯，Cross-attention让同一物点在不同视图的特征“互相对焦”，从而教网络“这些 patch 来自同一三维点”，为后面的匹配/深度回归提供强先验。

**直观理解**：

| 子模块 | 具体做法 | 作用 / 论文依据 |
| --- | --- | --- |
| **特征输入** | 每张图先经 6 层浅 ResNet 下采到 ¼ 分辨率 (H/4 × W/4) | 提取初步局部纹理 |
| **堆叠结构** | **6 个 Transformer block**，每块顺序是  ① **Self-Attention**（只看本视图 patch） ② **Cross-View Attention**（把本视图 token 当 “query”，其他视图 token 当 “key / value”） | Self 部分巩固自身上下文；Cross 部分显式交换多视图信息 |
| **注意力窗** | 用 **Swin-Transformer** 的 2 × 2 局部窗口机制，对长图高效；窗口划分对所有视图一致，保证对齐 | 既保局部分辨率，又控制计算量 |
| **视图数适配** | Cross-attention 对“其余全部视图”做一次性聚合，权重矩阵与视图数量无关，可在推理时随意增减输入视图 | 训练 2-view，推理 N-view 也能用 |
| **下游传播** | 得到的跨视一致特征 **Fi** 会与各自的 cost volume 拼接送入 U-Net；U-Net 内部最低分辨率层再插入 3 个 cross-view attention 层细化体数据 | 把跨视信息继续输往深度估计和高斯属性预测 |
| **有效性** | 去掉 cross-view attention，PSNR ↓1 dB、LPIPS↑，且训练过拟合更严重 | 证明跨视信息流动是几何学习关键 |

### 1. Multi-View Transformer（带跨视注意力）的细节

### L1 和 L2 损失的区别与作用

这两种损失函数都属于「重建类损失」，在深度学习中用来衡量预测值和真实值之间的差异，尤其常用于图像生成、回归任务（如重建深度图、RGB图）中。

---

### 🔹 **L1 Loss**（绝对误差）

**定义**：

L1=1N∑i=1N∣y^i−yi∣L_1 = \frac{1}{N} \sum_{i=1}^{N} \left| \hat{y}_i - y_i \right|

**直观含义**：每个像素预测错多少就罚多少（线性增长）

**优点**：

- 更鲁棒于噪声和 outlier；
- 更保边缘（因为梯度恒定）；

**缺点**：

- 不可导于 0 处（但可用伪导数解决）；
- 收敛速度可能慢。

---

### 🔹 **L2 Loss**（平方误差）

**定义**：

L2=1N∑i=1N(y^i−yi)2L_2 = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2

**直观含义**：预测越错，惩罚越大（平方增长）

**优点**：

- 对小误差收敛快；
- 可导且梯度平滑，优化稳定；

**缺点**：

- 对异常值非常敏感（大误差平方后变很大）；
- 容易模糊图像（因为更追求整体均匀）

---

### 📌 在 MVSplat 或图像生成中的选择：

| 场景 | 常用损失 | 原因 |
| --- | --- | --- |
| 渲染图与真实图的像素对比 | 通常 **L2 或 L1 + LPIPS** | L2 快速收敛，LPIPS 保结构纹理 |
| 边缘锐化、更注重轮廓 | 偏向 **L1** | 抑制图像模糊，更适合重建边界清晰图 |
| 存在较多 outlier（遮挡、运动） | L1 更鲁棒 | L2 会放大错误点的影响 |

---

### 总结对比：

| 比较项 | L1 | L2 |
| --- | --- | --- |
| 损失增长 | 线性 | 平方 |
| 对 outlier | 稳定 | 敏感 |
| 收敛速度 | 稍慢 | 快 |
| 图像效果 | 较清晰 | 易模糊 |
| 是否可导 | 不可导于 0（可近似） | 可导且平滑 |

> 所以实际工程中常常 L1 + LPIPS / L2 + LPIPS 混合使用，以平衡结构保持和色彩匹配，MVSplat 就用了后者。
>