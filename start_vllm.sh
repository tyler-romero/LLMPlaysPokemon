#!/bin/bash
# Start a vLLM server with OpenAI-compatible API endpoints

# Default values
MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL=2
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.9
QUANTIZATION=""
TOOL_CALL_PARSER="hermes"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) TENSOR_PARALLEL="$2"; shift 2 ;;
    --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --quantize) QUANTIZATION="$2"; shift 2 ;;
    --tool-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
    --help)
      echo "Usage: $0 --model MODEL [--host HOST] [--port PORT] [--gpus NUM] [--max-len NUM] [--gpu-util NUM] [--quantize TYPE] [--tool-parser PARSER]"
      echo "Example: $0 --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --gpus 2 --tool-parser hermes"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Build command
CMD="uv run --with vllm --with setuptools vllm serve"
CMD="$CMD $MODEL --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL"
CMD="$CMD --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL"
CMD="$CMD --task generate --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER"
[[ -n "$QUANTIZATION" ]] && CMD="$CMD --quantization $QUANTIZATION"

# Print info
echo "Starting vLLM server with model: $MODEL on $HOST:$PORT"
echo "API will be available at: http://$HOST:$PORT/v1"
echo "Running: $CMD"

# Execute command
exec $CMD