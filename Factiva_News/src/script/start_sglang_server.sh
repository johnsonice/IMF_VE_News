
#!/bin/bash

# Fixed values for server configuration

# engine params : https://docs.sglang.ai/backend/server_arguments.html


PORT=8101
API_KEY="abc"

MODEL_PATH="/ephemeral/home/xiong/data/hf_cache/Qwen/Qwen3-8B"
SERVED_MODEL_NAME="Qwen/Qwen3-8B"
GPU_DEVICE="0,1,2,3"
REASONING_PARSER="qwen3"
#TP_SIZE=4
DP_SIZE=2

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Start SGLang server
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --api-key "$API_KEY" \
    --context-length 8192 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --allow-auto-truncate \
    --constrained-json-whitespace-pattern "[\n\t ]*" \
    --dp-size $DP_SIZE \
    --reasoning-parser $REASONING_PARSER
    
    #--tp $TP_SIZE
    #--allow-auto-truncate #Allow automatically truncating requests that exceed the maximum input length instead of returning an error.



## some issues reported : https://github.com/sgl-project/sglang/issues/2216