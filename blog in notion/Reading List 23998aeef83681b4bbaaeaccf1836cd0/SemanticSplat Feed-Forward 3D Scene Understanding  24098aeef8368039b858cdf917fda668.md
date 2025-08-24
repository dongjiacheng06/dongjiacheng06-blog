# SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields

Status: To read

## 问题/背景：

---

- 传统 3D 场景理解通常按 **SfM → MVS → 语义标注** 三段流水线完成，误差易层层放大且依赖稠密多视图，难以在真实场景落地，尤其缺乏开放词表能力 。
- **SemanticSplat** 提出一种 *feed-forward*（一次前向即可推理）的 3D 语义高斯重建框架，将 **几何-外观-语义** 融合到单一显式表示中：
    1. 以 **语义各向异性高斯 (semantic anisotropic Gaussians)** 扩展 3D Gaussian Splatting；
    2. 用 **多条件特征融合**（SAM + CLIP-LSeg + 多视几何成本体）提升跨视角一致性；
    3. 采用 **两阶段蒸馏** 同时在 3D 中重建 *分割特征场* 与 *语言特征场*，支持点提示与开放词表分割 。

---

## 方法

---

![image.png](SemanticSplat%20Feed-Forward%203D%20Scene%20Understanding%20%2024098aeef8368039b858cdf917fda668/image.png)

### 1 高效深度估计

- 对每张输入图像构建 **Plane-Sweep Cost Volume**，再用 2D U-Net 回归视图一致深度，兼顾遮挡与几何细节 。

### 2 多条件语义特征融合

- 将 **SAM 编码特征** (分割语义) 与 **CLIP-LSeg 特征** (语言语义) 上采并拼接到 Cost Volume，联合 Transformer 特征回归统一潜在特征图 。

### 3 语义各向异性高斯

- 在传统高斯参数 (μ, α, Σ, c) 外，再追加语义向量 f，实现同时渲染 **RGB 图 C** 与 **语义特征图 F**；渲染公式见论文式 (1) 。
- 初始化：用深度反投影生成中心；随后并行预测颜色、协方差与语义属性，并通过两阶段蒸馏校正 。

### 4 两阶段语义蒸馏

1. **分割特征场蒸馏**：将高斯语义 rasterize 回 2D，与 SAM 编码特征做余弦相似度监督，并用 Focal + Dice Loss 保持 mask 提示兼容 。
2. **语言特征场蒸馏**：冻结分割分支后，再对齐 CLIP-LSeg 特征，并用 **Hierarchical-Context-Aware Pooling** 在多尺度 SAM mask 内平均，保证细粒度一致性。

### 5 总体损失

- **Photometric** (L1 + LPIPS)、**SAM 对齐**、**CLIP-LSeg 对齐** 三项共同优化，Mask Loss 权重 λ_mask=0.2 。

---

## 实验

---

---

## 贡献/成果

---

| # | 贡献点 | 解释 |
| --- | --- | --- |
| **1. Feed-forward 整体场景理解框架** | 作者把深度估计、显式 3D 高斯表示和语义建模整合成一次前向推理流程，预测 **“语义各向异性高斯”** 五元组 (μ, α, Σ, c, f)，从而 **同时** 输出几何、外观和多模态语义，无需逐场景优化，保持 3DGS 的实时效率。 |  |
| **2. 多条件特征融合（Multi-Conditioned Feature Fusion）** | 在传统 plane-sweep cost volume 里，**首次把 SAM 分割特征 + CLIP-LSeg 语言特征与几何相似度一起拼接**，再用轻量 2D U-Net 统一正则化。这样深度预测阶段就注入了分割边界和开放词表语义，显著提高跨视一致性与语义准确性。 |  |
| **3. 两阶段语义蒸馏（Segmentation➝Language）** | 提出 **先用 SAM 对齐分割特征场，再冻结分支、利用 CLIP-LSeg 细化语言特征场** 的“双阶段”策略，解决 2D 语义在跨视投影时的不一致问题，使高斯语义既支持点/框提示分割，又能开放词表检索。 |  |

**补充亮点**

- **语义各向异性高斯表示**：在传统 3DGS 四元组中新增语义向量 *f*，渲染时用同一透明度/核权重同步生成 RGB 图与语义特征图，实现像素级对齐。
- **轻量 2D U-Net + Transformer**：深度维当通道，配合跨视 Swin-Transformer 抽特征，兼顾全局上下文与资源占用，保持推理 ≈ 0.1 s。

> 一句话总结：SemanticSplat 在 3D Gaussian Splatting 框架内，用一次前向就把几何、外观和可提示/开放的语义一起建好，并通过 SAM + CLIP-LSeg 的双模融合与两阶段蒸馏，解决了以往 2D→3D 语义不一致与必须 per-scene 优化的难题。
> 

---

| 阶段 | 输入 / 输出 | 关键操作 | 目的 |
| --- | --- | --- | --- |
| **0. 任务与输入** | N 张稀疏视图 I_i + 相机矩阵 P_i | —— | 同时重建几何、外观**和**多模态语义（分割 + 语言） |
| **1. 多视图特征抽取** | I_i → F_i ∈ R^(H_s×W_s×C) | CNN 下采样 → Swin-Transformer**跨视交叉注意** | 获得跨视一致的 2D 特征，为后续匹配做准备 |
| **2. Plane-sweep Cost Volume + 深度回归** | F_i → Cost volume C_i ∈ R^(H_s×W_s×D) | 对每视图在 D 个深度候选上**重投影**其他视图特征，计算相关性；Cost volume ⊕ Transformer 特征送入轻量 2D U-Net + softmax → 深度图 | 生成**稠密、跨视一致**深度，为 3D 高斯奠基 |
| **3. 语义特征注入与统一潜在图** | SAM 特征 F^SAM_i, CLIP-LSeg 特征 F^LSeg_i | 双线性插值到 H_s×W_s → 按通道维与 C_i**拼接**→ 再过同一个 2D U-Net | 在深度估计阶段就把**分割语义 + 语言语义**融入几何流，得到融合 latent 特征图 |
| **4. 高斯中心初始化** | 深度图 → 3D 点云 | 利用相机矩阵**反投影**像素 → 作为高斯中心 μ_j | 无需逐场景优化即可一次前向获得几何 |
| **5. 高斯属性预测** | 图像特征 + Cost volume | 两个卷积层预测**不透明度 α、协方差 Σ、颜色 c**； 独立 Head 结合 latent 图 & 深度回归**语义向量 f** | 构成*(μ, α, Σ, c, f)*五元组 |
| **6. 语义各向异性高斯表示** | *(μ, α, Σ, c, f)* | 渲染时共用 α 与投影核 G_i(X)：
RGB C = Σ_i c_i·α_i·G_i·∏_(j<i)(1-α_j·G_j)
语义 F = Σ_i f_i·α_i·G_i·∏_(j<i)(1-α_j·G_j) | 颜色与语义**天然对齐**，可同时出图 |
| **7. 两阶段语义蒸馏** | 3D 高斯 ↔ 2D VFM | **(a) 分割场蒸馏**：将高斯语义 → 2D，与 SAM 编码对齐（余弦 + Focal + Dice）；**(b) 语言场蒸馏**：冻结分割分支，RAS 高斯语言向量，与 CLIP-LSeg 特征对齐（层次池化 + 余弦） | 解决 2D 语义跨视不一致，获得稳健**promptable & open-vocabulary**语义场 |
| **8. 训练目标** | Photometric (L1 + LPIPS) + 分割蒸馏损失 + 语言蒸馏损失 | 端到端优化五元组参数 | 几何、外观、语义协同收敛 |
| **9. 推理与应用** | 单次前向（≈0.1 s） | → 新视角渲染（RGB + 深度 + 语义）→ 点提示分割、开放词表分割…… | 在 AR、机器人等场景中实时执行 |

下面先把「**传统 MVS 里 plane-sweep cost volume + U-Net**」这件事讲清楚，然后列出几条已经发表（比 SemanticSplat 更早或同时代）的代表性工作，方便你快速定位文献。

---

## 1 · 为什么要用 plane-sweep cost volume？

- **Plane-sweep** 本质：在参考相机前沿着深度方向切 D 张平面，把其他视图的特征都 **重投影** 到这些平面上，计算相似度，存成一个三维张量（横向 H × 纵向 W × 深度层 D）。
- 最早可追溯到 Collins 1996 的多视图匹配算法——今天深度网络沿用的仍是这一思想。([ri.cmu.edu](https://www.ri.cmu.edu/pub_files/pub1/collins_robert_1996_1/collins_robert_1996_1.pdf?utm_source=chatgpt.com))

---

## 2 · 典型流水线（以 MVSNet 家族为例）

| 步骤 | 传统做法 | 所用网络 |
| --- | --- | --- |
| **特征提取** | 对每张图跑 CNN(FPN/ResNet) 得到 1/4 ～ 1/8 分辨率的特征图 | 2D CNN |
| **构建 cost volume** | *对每一个深度平面*：把所有源视图特征 **单应变换（homography warp）** 到参考视图 → 计算方差 / 相关性 → 汇聚多视图 | 无学习或很浅的 2D op |
| **Regularization** | 把 H × W × D × C 的体数据丢进 **U-Net 形式的卷积网络**，在体内做空间-深度信息传播、降噪、补洞 | *两种分支：* • **3D U-Net**（3D 卷积，效果好但吃显存）• **2D U-Net**（把 D 当通道，用2D卷积，省显存） |
| **深度回归** | 在 regularized volume 上做 soft-argmax / softmax，得到概率体 → 求期望或取 argmax 得到深度图 | 无参数或单层 softmax |

> 这一套“plane-sweep → cost volume → U-Net regularization → softmax 回归”的管线，基本成了 2018 年以来深度 MVS 的默认模版。
> 

---

## 3 · 已经用过这套做法的代表论文

| 论文 & 发表时间 | plane-sweep 细节 | Regularization 网络 | 亮点 | 参考 |
| --- | --- | --- | --- | --- |
| **MVSNet** (ECCV 2018) | 可微单应变换生成 cost volume | **3D U-Net**（encoder-decoder，带跳连） | 第一次把 plane-sweep 完整搬进端到端网络，在 DTU/T&T 大幅刷新 SOTA | ([CVF开放获取](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf)) |
| **MVDepthNet** (3DV 2018) | 直接用像素差构造 cost volume | **轻量 2D U-Net**（encoder-decoder）将 D 当通道，实时运行 | 牺牲少量精度换实时速度，更易嵌入 SLAM / AR | ([ResearchGate](https://www.researchgate.net/publication/328312320_MVDepthNet_Real-Time_Multiview_Depth_Estimation_Neural_Network)) |
| **R-MVSNet** (CVPR 2019) | 与 MVSNet 同样的 plane-sweep | 用 **GRU 沿深度递归**，一层一层扫，显存更省 | 支持 3072 px 超高分辨率输入 | ([CVF开放获取](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Recurrent_MVSNet_for_High-Resolution_Multi-View_Stereo_Depth_Inference_CVPR_2019_paper.pdf?utm_source=chatgpt.com)) |
| **CasMVSNet** (CVPR 2020) | 多尺度 plane-sweep，粗到细逐级细化 | 每一级一个 **小 3D U-Net** | 速度-精度折衷优秀，之后被很多方法沿用 | ([GitHub](https://github.com/kwea123/CasMVSNet_pl)) |
| **HR-MVSNet / CVP-MVSNet / PatchMatchNet / CT-MVSNet**（2021-2023） | 在 plane-sweep 上加 **金字塔、Transformer、PatchMatch Propagation** 等技巧 | 2D 或 3D U-Net、Transformer、调制卷积等 | 目标都是在精度、显存、速度之间再做权衡 | ([bmvc2022.mpi-inf.mpg.de](https://bmvc2022.mpi-inf.mpg.de/0073.pdf?utm_source=chatgpt.com), [arXiv](https://arxiv.org/html/2312.08594v2?utm_source=chatgpt.com)) |

> 这些工作已经证明：
> 
> 
> • **U-Net**（不论 2D 还是 3D） 对 cost volume 的“平滑+上下文传递”非常有效；
> 
> • plane-sweep cost volume + U-Net 是社区最常用、也最稳妥的深度 MVS 策略之一。
> 

---

## 4 · 和 SemanticSplat 的差别

- **几何部分**：SemanticSplat 直接拿 **MVSplat** (CVPR 2024) 那一版 plane-sweep 2D U-Net 做深度；在“几何推理” 上确实是沿用已有套路。
- **创新点** 在于：
    1. **把 SAM 分割特征和 CLIP-LSeg 语言特征也拼进了 cost volume**，让深度网络自带语义先验；
    2. 后端不是输出纯深度，而是直接生成带 *语义向量 f* 的 3D Gaussians，并用两阶段蒸馏对齐 2D-3D 语义。

---

### 小结

- **Plane-sweep cost volume + U-Net** 早就是深度 MVS 的“祖传套路”（最早可追溯到 2018 年的 MVSNet）。
- 此后的 R-MVSNet、CasMVSNet、MVDepthNet 等都在同一框架下改进显存或速度。
- SemanticSplat 并不是在这个环节上推新算法，而是 **继承成熟几何模块，把语义特征和高斯表示整合进来**，把重点放到了“几何-外观-语义一体化”上。

### “把深度层当 channel 的 2D U-Net” vs “真正的 3D U-Net”

| 对比点 | 2D U-Net：把深度层拼到通道维 | 3D U-Net：用三维卷积核滑动 |
| --- | --- | --- |
| **卷积核形状** | 每层卷积核大小是 *(kh × kw × C)*，只在**平面**内移动；深度方向当作普通通道，仅在同一平面位置做一次性加权混合 | 卷积核大小是 *(kh × kw × kd × cin)*，会在 **x、y、z** 三维滑动，沿深度维逐格卷积 |
| **信息传播方式** | 不会向“邻近深度层 + 邻近像素”扩散；只能让**同一像素坐标**但不同深度层之间相互影响 | 每次卷积都同时考虑“平面邻域 + 深度前后邻域”，邻近体素会互相传递特征 |
| **深度维的平移等变性** | 内部权重**不会重复使用**到相邻深度层（因为深度被当成固定通道索引），没有“沿深度的卷积平移不变性” | 卷积核在深度轴上滑动，天然具有 **沿深度轴的平移不变性** |
| **可表达的 3D 结构** | 只能学“在同一像素上、不同深度层的组合关系”，缺乏对体块或斜坡状结构的建模能力 | 能直接捕获 3D 连通体、倾斜面、连续表面等立体模式 |
| **显存与算力** | 与深度层数成**线性**关系；因为仍是 2D 卷积，计算量较小 | 与深度层数成**立方**/平方级增长（取决于 kd），显存和算力显著更高 |
| **典型用法** | 深度层很多、显存紧张；需要快速推理或在 GPU 少的环境 | 医学 CT/MRI、稠密体素、MVS cost volume 尺寸较小但要求 3D 连贯性强 |
| **常见问题** | - 无法捕捉跨深度的平移模式- 需要更多层堆叠才能间接传递 3D 语义 | - 显存大、训练慢- 当深度维非常大时必须切 patch 或降采样 |

---

### 用一句话归纳

- **2D 方法把深度当通道**：像一次把所有深度切片叠成彩色图，只在每个平面上卷积；它**看得到同坐标的全部深度信息**，但**看不到相邻深度像素**的空间关系。
- **3D U-Net**：卷积核在三维体里滑动，**既看同坐标、又看前后邻层与周围邻层**，能直接学习 3D 形状，但显存和算力开销更大。

所以如果你关心三维连贯性（器官边界、稠密深度的平滑）且显存允许，就用 3D U-Net；如果更关注速度或数据体很“扁”（深度维巨大），就把深度当通道用 2D U-Net 并配合金字塔 / Patch / GRU 等技巧来弥补信息传播不足。

用的就是 **把 D 个深度层当作通道的 2D U-Net**，而不是真正的 3D U-Net。

论文在两处写得很清楚：

- “最后由 **2D U-Net** + softmax 预测每视图深度图”
- 在把 SAM / CLIP-LSeg 特征拼进 cost volume 后，又交给 “**lightweight 2D U-Net**” 做融合

也就是说，它仅在 H×W 平面做卷积，深度维只是额外的通道，因而显存和计算量都比 3D 卷积轻得多。

**论文里的 Transformer 在哪里用？**

| 位置 | 做了什么 | 目的 |
| --- | --- | --- |
| **Multi-View Feature Extraction** | 先用 CNN 下采样各视图特征，再送进一段 **带跨视交叉注意力的 Swin-Transformer**。该模块把不同相机之间的特征互相“看一眼”，输出跨视一致的特征 Fi【71-77】 | ① 在纹理贫乏或遮挡场景里补足信息 ② 为后续深度匹配提供“已经对齐”的特征基础 |
| **Depth Regression** | plane-sweep cost volume Ci 与上面得到的 Transformer 特征 **按通道拼接**，一起喂给轻量 2D U-Net；U-Net + softmax 输出稠密深度图【69-71】 | 进一步利用全局上下文，减少噪声，提升边缘/遮挡处深度一致性 |
| **语义特征融合** | 拼接了 SAM 和 CLIP-LSeg 特征后，依旧用 **同一个 2D U-Net** 做正则化；Transformer 特征作为几何“主干”，语义特征提供额外通道 | 让网络一次前向就同时考虑几何与语义，避免后期再对齐 |

---

### 目的总结

- **跨视信息传播**：交叉注意力层让每个视图都能借到其他视图的细节，解决纹理重复 / 视角遮挡带来的匹配困难。
- **减少噪声**：相比单纯 CNN 或局部 3D 卷积，Transformer 具有更大感受野与自适应权重，能在 cost volume 进入 U-Net 之前先“滤波”。
- **保持实时**：只在 H×W 平面跑自注意力，深度维仍当通道，计算量可控。

---

### 是否原创？

- **并非首创**：跨视 Transformer 在 MVS 场景早有 **TransMVSNet、MVSplat** 等工作；SemanticSplat 直接沿用了 MVSplat 的实现，并在论文中说明“Following MVSplat …”【13-17】 。
- **真正的创新** 在于：在同一管线里把 **SAM 分割特征、CLIP-LSeg 语言特征** 注入 cost volume，并把输出直接变成带语义向量 f 的高斯五元组，实现了“几何-外观-语义一次成型”。Transformer 模块只是配合这条语义增强通路，用来稳固几何基础。

**论文同时使用 SAM 和 CLIP-LSeg 的原因，可以用一句话概括：**

> SAM 负责“像素级边界 + 可提示分割”，CLIP-LSeg 负责“开放词表的语义标签”，两者互补，共同保证 3D 语义场既精细又可自由命名。
> 

---

### 1. 两个 VFM 的优势与短板

| 模型 | 优势 | 局限 |
| --- | --- | --- |
| **SAM** | *分割质量高*：给定点 / 框就能输出非常干净的实例或区域 mask；*几何感强*，边界对齐像素 | **不懂语义**：只能切出“物体轮廓”，但不知道它叫什么 |
| **CLIP-LSeg** | *语言对齐*：特征空间直接和文本嵌入相匹配，天生支持开放词表类别；*可一次输出全景语义图* | 掩膜粗、边缘毛糙，且跨视图一致性差 |

作者观察到，两者正好“一补一缺”。于是整条管线把 **SAM → 保形状，CLIP-LSeg → 给名字** 结合：

- **特征级融合**：首先把两种 2D 语义特征上采到与 cost-volume 相同分辨率，然后与几何 cost-volume **按通道拼接**，交给轻量 2D U-Net，一次性学到“几何 + 分割 + 语言”统一潜在特征
- **两阶段蒸馏**
    1. **阶段 ①：SAM 蒸馏**
        
        用 SAM 编码特征指导 Gaussian 语义向量，外加 Focal+Dice loss，先确保 3D 场与 2D 掩膜一致，可被点提示调用
        
    2. **阶段 ②：CLIP-LSeg 蒸馏**
        
        冻结分割分支后，再对齐 CLIP-LSeg 特征；通过层次掩膜池化，让语言向量在不同尺度上保持一致，获得真正的开放词表能力
        

这样做带来三点好处：

1. **跨视一致性**：分割边界先被 SAM 约束，再由 CLIP 语义细化，显著减少因为视角变化导致的标签抖动
2. **双模式输出**：同一套高斯既能被点 / 框 Prompt 成实例 mask，也能直接按文本查询任意类别，实现“一次建场，多种用法”。
3. **无需两遍推理**：把两种特征都塞进 cost-volume 和同一个 U-Net，保持全流程 **feed-forward**，推理时间 ≈ 0.1 s

---

### 2. 是否原创？

- **多条件语义特征融合**（SAM + CLIP-LSeg + cost-volume）被作者列为贡献 ②。
- 早期 3DGS 扩展里通常 **只用 SAM** 或 **只用 CLIP**；同时利用两者并保持端到端推理，在目前公开文献中属较新做法。

---

### 3. 一句话总结

> SemanticSplat 需要 SAM 来保证分割轮廓干净、支持点提示，需要 CLIP-LSeg 来注入可搜索的语言语义；两者在 cost-volume 和两阶段蒸馏里协同，使得最终 3D 场景既“形准”又“名正”，这正是论文的第二大贡献。
>