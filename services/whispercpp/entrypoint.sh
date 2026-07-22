#!/usr/bin/env bash
set -euo pipefail

model_name="${WHISPERCPP_MODEL:-small}"
model_dir="${WHISPERCPP_MODEL_DIR:-/models}"
model_path="${WHISPERCPP_MODEL_PATH:-${model_dir}/ggml-${model_name}.bin}"
model_url="${WHISPERCPP_MODEL_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${model_name}.bin}"

mkdir -p "${model_dir}"
if [[ ! -s "${model_path}" ]]; then
  echo "Downloading whisper.cpp model ${model_name}..."
  temporary_path="${model_path}.download"
  rm -f "${temporary_path}"
  curl --fail --location --retry 4 --retry-all-errors \
    --output "${temporary_path}" "${model_url}"
  mv "${temporary_path}" "${model_path}"
fi

arguments=(
  --host "0.0.0.0"
  --port "${WHISPERCPP_PORT:-8080}"
  --model "${model_path}"
  --language "${WHISPERCPP_LANGUAGE:-auto}"
  --threads "${WHISPERCPP_THREADS:-6}"
  --suppress-nst
)

if [[ "${WHISPERCPP_USE_GPU:-false}" != "true" ]]; then
  arguments+=(--no-gpu)
fi
if [[ "${WHISPERCPP_FLASH_ATTENTION:-false}" == "true" ]]; then
  arguments+=(--flash-attn)
fi

exec /usr/local/bin/whisper-server "${arguments[@]}" "$@"
