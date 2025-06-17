#!/bin/bash

# Define paths to your scripts
STYLE_TRANSFER_SCRIPT="25-1-TF-style-transfer-module.py"
RIFE_SCRIPT="25-2-RIFE-module.py"

# Activate style transfer environment
echo "🔁 Activating style transfer Conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env

# Run style transfer
echo "🎨 Running style transfer..."
python3 "$STYLE_TRANSFER_SCRIPT"
if [ $? -ne 0 ]; then
    echo "❌ Style transfer failed. Exiting."
    exit 1
fi

# Activate RIFE environment
echo "🔁 Activating RIFE Conda environment..."
conda activate rife

# Run RIFE frame interpolation
echo "🎬 Running RIFE interpolation..."
python3 "$RIFE_SCRIPT"
if [ $? -ne 0 ]; then
    echo "❌ RIFE interpolation failed. Exiting."
    exit 1
fi

echo "✅ All steps completed successfully!"
