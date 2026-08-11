#!/usr/bin/env bash
# One-time setup for Mac (M-series or Intel).
# Run once from the project root: bash setup.sh

set -e
echo "=== S&P 500 Forecasting Pipeline — Mac Setup ==="

# 1. Python version check
python3 --version | grep -E "3\.(10|11|12)" || {
    echo "❌  Python 3.10, 3.11 or 3.12 required. Install via: brew install python@3.12"
    exit 1
}

# 2. Install dependencies
echo ""
echo "Installing Python packages..."
pip3 install -r requirements.txt

# 3. PyTorch — install CPU/MPS version if not already present
python3 -c "import torch; print(f'✓  PyTorch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# 4. Verify CRSP data exists
if [ ! -f "data/processed/crsp_sp500_daily_corrected_2010_2024.parquet" ]; then
    echo ""
    echo "⚠️   CRSP data not found at data/processed/crsp_sp500_daily_corrected_2010_2024.parquet"
    echo "    Download from WRDS and place it there before running the pipeline."
    echo "    See README.md for instructions."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To smoke-test the pipeline in seconds (tiny universe/window, no real results):"
echo "    python run.py --quick"
echo ""
echo "To run the full pipeline (multi-hour — step2/step3 are genuinely walk-forward now,"
echo "see README.md 'Runtime' section):"
echo "    python run.py"
echo ""
echo "To skip TFT (faster, 5-model results):"
echo "    python run.py --skip-tft"
echo ""
echo "To run a single step:"
echo "    python run.py --step 1   # classical only"
echo "    python run.py --step 4   # downstream only (after steps 1-3 done)"
