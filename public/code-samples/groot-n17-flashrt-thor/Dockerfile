# Build with the pinned FlashRT image as BASE_IMAGE and the harness scripts/ directory as context.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglfw3 libxrandr2 \
    libxinerama1 libxcursor1 libxi6 && rm -rf /var/lib/apt/lists/*
COPY install_libero_runtime.py /opt/libero/install_runtime.py
COPY libero-config.yaml /opt/libero/config.yaml
RUN /opt/reference/groot-venv/bin/python /opt/libero/install_runtime.py
RUN /opt/reference/groot-venv/bin/python -m pip install --no-cache-dir --no-deps \
    gymnasium==1.2.2 farama-notifications==0.0.6 \
    nvidia-modelopt==0.39.0 onnx==1.19.1 torchprofile==0.1.0 \
    onnx-graphsurgeon==0.6.1 onnxconverter-common==1.16.0 onnxruntime==1.22.1 \
    onnxscript==0.6.2 onnx-ir==0.2.0 polygraphy==0.53.6 \
    coloredlogs==15.0.1 flatbuffers==25.12.19 humanfriendly==10.0 \
    cppimport==26.4.17 mako==1.3.10 pybind11==3.0.1
ENV LIBERO_CONFIG_PATH=/opt/libero \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    PYTHONPATH=/comparison/vendor/Isaac-GR00T:/comparison/vendor/Isaac-GR00T/external_dependencies/LIBERO:/comparison:/opt/FlashRT \
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1
WORKDIR /comparison
ENTRYPOINT ["/opt/reference/groot-venv/bin/python"]
