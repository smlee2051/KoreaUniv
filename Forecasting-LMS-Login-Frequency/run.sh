#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh [GROUP_ID] [MODEL_NAME]"
    echo ""
    echo "GROUP_ID options:"
    echo "  1 = models_group1 (Transformer, Reformer, Informer, Autoformer)"
    echo "  2 = models_group2 (LLMTime GPT-3.5, LLMTime GPT-4o, PromptCast GPT-3.5)"
    echo "  3 = models_group3 (LLAMA, GPT2, BERT)"
    echo ""
    echo "MODEL_NAME examples:"
    echo "  ./run.sh 1 Autoformer"
    echo "  ./run.sh 2 \"LLMTime GPT-3.5\""
    echo "  ./run.sh 3 LLAMA"
    exit 1
fi

GROUP_ID=$1
MODEL_NAME="${@:2}"

ENV_DIR="/path/to/envs"

if [ "$GROUP_ID" == "1" ]; then
    echo "========== Running Group 1: models_group1 =========="
    cd models_group1
    source "$ENV_DIR/models_group1_env/bin/activate"
    echo "Running model: $MODEL_NAME"
    python main.py "$MODEL_NAME"
    deactivate
    cd ..

elif [ "$GROUP_ID" == "2" ]; then
    echo "========== Running Group 2: models_group2 =========="
    cd models_group2
    source "$ENV_DIR/models_group2_env/bin/activate"
    echo "Running model: $MODEL_NAME"
    python main.py "$MODEL_NAME"
    deactivate
    cd ..

elif [ "$GROUP_ID" == "3" ]; then
    echo "========== Running Group 3: models_group3 =========="
    cd models_group3
    source "$ENV_DIR/models_group3_env/bin/activate"
    echo "Running LLM model: $MODEL_NAME"
    python main.py "$MODEL_NAME"
    deactivate
    cd ..

else
    echo "Invalid GROUP_ID: $GROUP_ID"
    echo "Valid options are 1, 2, or 3"
    exit 1
fi
