---
title: 3D Gaussian Splatting Windows 部署完整指南
date: 2025-01-22 10:00:00 +0800
categories: [3D Vision, Tutorial]
tags: [3DGS, Windows, CUDA, PyTorch, 环境配置, 部署指南]
math: true
mermaid: true
image:
  path: /assets/images/3dgs-deployment.png
  alt: 3D Gaussian Splatting Windows 部署
---

## 概述

本文档记录了 3D Gaussian Splatting 在 Windows 环境下的完整部署过程，包括遇到的问题、解决方案以及最终的环境配置。

## 系统环境

### 硬件配置
- **操作系统**: Windows 11
- **GPU**: NVIDIA GPU (支持 CUDA Compute Capability 7.0+)
- **内存**: 建议 16GB+ (训练需要大量内存)
- **存储**: 建议 SSD，至少 20GB 可用空间

### 软件版本
- **Python**: 3.8.20
- **CUDA**: 11.8
- **PyTorch**: 2.0.0+cu118
- **Visual Studio**: 2019 Community (14.29.30133)
- **Anaconda**: 最新版本
- **Git**: 最新版本

## 部署步骤

### 第一步：环境准备

#### 1.1 安装 CUDA 11.8
```bash
# 下载并安装 CUDA 11.8
https://developer.nvidia.com/cuda-11-8-0-download-archive
```

#### 1.2 安装 Visual Studio 2019 Community
```bash
# 下载地址
https://visualstudio.microsoft.com/vs/older-downloads/
# 确保安装 C++ 构建工具和 Windows SDK
```

#### 1.3 验证环境
```powershell
# 验证 CUDA 安装
nvcc --version

# 验证 Visual Studio 编译器
where cl.exe
```

### 第二步：克隆项目

```bash
# 克隆 3D Gaussian Splatting 仓库
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
```

### 第三步：创建虚拟环境

```bash
# 创建 conda 环境
conda create -n gaussian_splatting python=3.8
conda activate gaussian_splatting
```

### 第四步：安装依赖

#### 4.1 安装 PyTorch
```bash
# 安装 PyTorch 2.0.0 + CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 4.2 安装其他依赖
```bash
# 安装必要的 Python 包
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## 常见问题与解决方案

### 问题1：Visual Studio 版本冲突

**错误信息**：
```
Microsoft Visual C++ 14.0 is required
```

**解决方案**：
```bash
# 确保安装了正确版本的 Visual Studio 2019
# 并且在环境变量中设置了正确的路径
set "VS160COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\"
```

### 问题2：CUDA 路径问题

**错误信息**：
```
CUDA_HOME environment variable is not set
```

**解决方案**：
```bash
# 设置 CUDA 环境变量
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_HOME%\bin;%PATH%
```

### 问题3：中文路径问题

**错误信息**：
```bash
UnicodeDecodeError: 'gbk' codec can't decode
```

**解决方案**：
- 确保项目路径中不包含中文字符
- 使用英文路径，如 `C:\Projects\gaussian-splatting`

### 问题4：GLM 库缺失

**错误信息**：
```
fatal error C1083: Cannot open include file: 'glm/glm.hpp'
```

**解决方案**：
```bash
# 手动下载并配置 GLM 库
# 1. 下载 GLM: https://github.com/g-truc/glm/releases
# 2. 解压到 C:\glm
# 3. 设置环境变量
set GLM_ROOT=C:\glm
```

## 验证安装

### 运行测试

```bash
# 下载测试数据集
# 运行训练脚本
python train.py -s <path_to_scene> -m <output_path>

# 运行推理脚本
python render.py -m <trained_model_path>
```

### 性能基准

在我们的测试环境中（RTX 3080，32GB RAM）：

| 场景 | 训练时间 | 渲染速度 | 内存占用 |
|------|----------|----------|----------|
| 室内小场景 | ~30分钟 | 60+ FPS | ~8GB |
| 室外大场景 | ~2小时 | 30+ FPS | ~16GB |

## 最佳实践

### 1. 环境管理
- 使用独立的 conda 环境避免包冲突
- 定期备份工作环境

### 2. 数据准备
- 确保输入图像质量良好
- 相机标定数据准确

### 3. 训练优化
- 根据 GPU 内存调整 batch size
- 使用混合精度训练节省内存

## 总结

通过以上步骤，你应该能够在 Windows 环境下成功部署 3D Gaussian Splatting。关键是确保所有依赖版本的兼容性，特别是 CUDA、PyTorch 和 Visual Studio 的版本匹配。

如果在部署过程中遇到其他问题，建议：
1. 检查官方 GitHub Issues
2. 确认硬件兼容性
3. 查看详细的错误日志

## 参考资料

- [3D Gaussian Splatting 官方仓库](https://github.com/graphdeco-inria/gaussian-splatting)
- [CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

---

*本指南基于实际部署经验总结，如有问题欢迎交流讨论！*
