# ============================================
# Base Image: PyTorch 2.1.0 + CUDA 12.1 (适配 RTX 4090)
# ============================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置环境变量：防止安装交互卡死，设置时区
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 优化 APT 源 (可选，国内服务器推荐开启，海外服务器请注释掉)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 1. 安装系统级依赖 (OpenCV, nuScenes, 编译工具必备)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 2. 复制依赖清单并安装
COPY requirements.txt /workspace/

# 3. 安装 Python 依赖
# 技巧：使用清华源加速，且不使用缓存减小体积
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 关键：安装适配 RTX 4090 的 spconv (稀疏卷积库)
# 必须使用 cu120 版本，否则 PointPillars 会报错
RUN pip install --no-cache-dir spconv-cu120 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装 OpenMMLab 基础库 (BEV-MAE 等算法常用)
RUN pip install --no-cache-dir openmim -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmdet3d>=1.0.0"

# 默认启动命令
CMD ["/bin/bash"]
