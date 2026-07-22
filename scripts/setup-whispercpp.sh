#!/usr/bin/env bash
set -euo pipefail

VERSION="${WHISPER_CPP_VERSION:-v1.8.5}"
MODEL="${WHISPER_CPP_MODEL:-small}"
ENABLE_CUDA="${WHISPER_CPP_CUDA:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${WHISPER_CPP_RUNTIME_DIR:-${ROOT_DIR}/runtime/whispercpp}"
SOURCE_DIR="${RUNTIME_DIR}/source"
BUILD_DIR="${SOURCE_DIR}/build"
BIN_DIR="${RUNTIME_DIR}/bin"
MODEL_DIR="${RUNTIME_DIR}/models"

mkdir -p "${RUNTIME_DIR}" "${BIN_DIR}" "${MODEL_DIR}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${VERSION}" \
    https://github.com/ggml-org/whisper.cpp.git "${SOURCE_DIR}"
else
  git -C "${SOURCE_DIR}" fetch --depth 1 origin "${VERSION}"
  git -C "${SOURCE_DIR}" checkout --force FETCH_HEAD
fi

cmake_args=(-S "${SOURCE_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release)
if [[ "${ENABLE_CUDA}" == "1" ]]; then
  cmake_args+=(-DGGML_CUDA=ON)
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --config Release --parallel
cp "${BUILD_DIR}/bin/whisper-cli" "${BIN_DIR}/whisper-cli"

(
  cd "${SOURCE_DIR}"
  bash ./models/download-ggml-model.sh "${MODEL}"
)
cp "${SOURCE_DIR}/models/ggml-${MODEL}.bin" "${MODEL_DIR}/ggml-${MODEL}.bin"

printf 'whisper.cpp %s installed\n' "${VERSION}"
printf 'binary: %s\n' "${BIN_DIR}/whisper-cli"
printf 'model:  %s\n' "${MODEL_DIR}/ggml-${MODEL}.bin"
