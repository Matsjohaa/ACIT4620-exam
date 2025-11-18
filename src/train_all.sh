#!/bin/bash
# Train all 7 zones one by one

source .venv/bin/activate

echo "================================================================================"
echo "TRAINING ALL 7 ZONES - Started at $(date)"
echo "================================================================================"

# Zone 1
echo ""
echo "[1/7] Training IT-NORD..."
python src/train.py --zones IT-NORD --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 2
echo ""
echo "[2/7] Training IT-CNOR..."
python src/train.py --zones IT-CNOR --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 3
echo ""
echo "[3/7] Training IT-CSUD..."
python src/train.py --zones IT-CSUD --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 4
echo ""
echo "[4/7] Training IT-SUD..."
python src/train.py --zones IT-SUD --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 5
echo ""
echo "[5/7] Training IT-SICI..."
python src/train.py --zones IT-SICI --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 6
echo ""
echo "[6/7] Training IT-SARD..."
python src/train.py --zones IT-SARD --epochs 25 --model-type encoder --residual --batch-size 32

# Zone 7
echo ""
echo "[7/7] Training IT-CALA..."
python src/train.py --zones IT-CALA --epochs 25 --model-type encoder --residual --batch-size 32

echo ""
echo "================================================================================"
echo "ALL 7 ZONES COMPLETE! - Finished at $(date)"
echo "================================================================================"
