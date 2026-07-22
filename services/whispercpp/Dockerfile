ARG WHISPER_CPP_VERSION=v1.8.5

FROM debian:bookworm-slim AS builder
ARG WHISPER_CPP_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${WHISPER_CPP_VERSION}" \
    https://github.com/ggml-org/whisper.cpp.git /src/whisper.cpp
RUN cmake -S /src/whisper.cpp -B /src/whisper.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build /src/whisper.cpp/build --config Release \
    --target whisper-server -j"$(nproc)"

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/whisper.cpp/build/bin/whisper-server /usr/local/bin/whisper-server
COPY entrypoint.sh /usr/local/bin/whispercpp-entrypoint
RUN chmod +x /usr/local/bin/whispercpp-entrypoint

ENV WHISPERCPP_MODEL=small \
    WHISPERCPP_MODEL_DIR=/models \
    WHISPERCPP_LANGUAGE=auto \
    WHISPERCPP_THREADS=6 \
    WHISPERCPP_USE_GPU=false \
    WHISPERCPP_FLASH_ATTENTION=false \
    WHISPERCPP_PORT=8080

VOLUME ["/models"]
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10m --retries=12 \
    CMD curl --fail --silent http://localhost:8080/health || exit 1
ENTRYPOINT ["/usr/local/bin/whispercpp-entrypoint"]
