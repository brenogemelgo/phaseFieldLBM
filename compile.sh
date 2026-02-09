#!/bin/bash
set -euo pipefail

GPU_INDEX=${GPU_INDEX:-0}
if [ -n "${GPU_ARCH:-}" ]; then
    CC="${GPU_ARCH}"
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_INDEX}" 2>/dev/null | head -n1 | tr -d '.')
    fi
    CC=${CC:-86} 
fi

FLOW_CASE=${1:-}
VELOCITY_SET=${2:-}
ID=${3:-}

if [ -z "$FLOW_CASE" ] || [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
    echo "Usage: ./compile.sh <FLOW_CASE> <VELOCITY_SET> <ID>"
    echo "Example: ./compile.sh JET D3Q19 000"
    exit 1
fi

if [ "$VELOCITY_SET" != "D3Q19" ] && [ "$VELOCITY_SET" != "D3Q27" ]; then
    echo "Invalid VELOCITY_SET. Use 'D3Q19' or 'D3Q27'."
    exit 1
fi

# Load velocity set into a compile flag macro after validation to avoid issues with classes
VS_MACRO="VS_${VELOCITY_SET}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: nvcc not found in PATH."
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -d "${SCRIPT_DIR}/src" ]; then
    BASE_DIR="${SCRIPT_DIR}"
elif [ -d "${SCRIPT_DIR}/../src" ]; then
    BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Error: could not locate project root (missing 'src/' next to or above compile.sh)."
    echo "Checked: ${SCRIPT_DIR}/src and ${SCRIPT_DIR}/../src"
    exit 1
fi

SRC_DIR="${BASE_DIR}/src"
OUTPUT_DIR="${BASE_DIR}/bin/${FLOW_CASE}/${VELOCITY_SET}"
EXECUTABLE="${OUTPUT_DIR}/${ID}sim_${FLOW_CASE}_${VELOCITY_SET}_sm${CC}"

mkdir -p "${OUTPUT_DIR}"

echo "Project root detected: ${BASE_DIR}" 
echo "Compiling to ${EXECUTABLE}..."

nvcc -O3 --restrict \
     -gencode arch=compute_${CC},code=sm_${CC} \
     -gencode arch=compute_${CC},code=lto_${CC} \
     -rdc=true \
     --ptxas-options=-v \
     --extra-device-vectorization \
     --fmad=true \
     --extended-lambda \
     -I"${SRC_DIR}" \
     -std=c++20 "${SRC_DIR}/main.cu" \
     -D${VS_MACRO} -D${FLOW_CASE} \
     -DENABLE_FP16=1 \
     -DBENCHMARK=0 \
     -DTIME_AVERAGE=1 \
     -DREYNOLDS_MOMENTS=1 \
     -DVORTICITY_FIELDS=1 \
     -DPASSIVE_SCALAR=0 \
     -o "${EXECUTABLE}"

echo "Compilation completed successfully: ${EXECUTABLE}"
