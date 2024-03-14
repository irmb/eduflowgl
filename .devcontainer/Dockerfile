
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04




WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglew-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libglfw3-dev \
    libsdl2-dev \
    libgl1-mesa-glx \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libsdl2-image-dev \
    cmake \
    # nvidia-utils-550 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY .. /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


ENV PATH="/usr/local/cuda/bin:${PATH}"


ENV CUDA_PATH="/usr/local/cuda"




