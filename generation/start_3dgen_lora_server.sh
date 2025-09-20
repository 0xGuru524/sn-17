#!/bin/bash
# Startup script for 3D Generation LoRA Server

echo "Starting 3D Generation LoRA Server..."

# Set environment variables
export HF_HOME=/nvme0n1-disk/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if LoRA file exists
LORA_PATH="3dgen-lora/weights/3dgen.safetensors"
if [ ! -f "$LORA_PATH" ]; then
    echo "⚠️  Warning: LoRA file not found at $LORA_PATH"
    echo "   Server will run without LoRA weights"
fi

# Start the server
echo "Starting server on port 10007..."
python serve_3dgen_lora.py --port 10007 --lora-path "$LORA_PATH"
