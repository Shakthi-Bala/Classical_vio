# Builds the C++ VIO and its tests on Ubuntu 24.04 (x86-64 or arm64).
#
#   docker build -t classical-vio -f cpp/Dockerfile .
#   docker run --rm -v /path/to/MH_01_easy:/data -v $PWD/out:/out classical-vio \
#       --path /data --config config/euroc_mono.yaml --backend swo --out /out
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build g++ libeigen3-dev libopencv-dev libceres-dev \
        libyaml-cpp-dev libboost-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

COPY cpp /vio/cpp
WORKDIR /vio/cpp
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure

ENTRYPOINT ["/vio/cpp/build/main_euroc"]
