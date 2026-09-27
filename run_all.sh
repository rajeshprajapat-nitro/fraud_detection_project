#!/bin/bash
# run_all.sh — One command to set up and run the ENTIRE project.
#
# Usage:
#   1. Put creditcard.csv inside the data/ folder (see README for download steps)
#   2. From the fraud_project/ folder, run:
#        bash run_all.sh
#
# This will: create a virtual environment, install dependencies,
# run EDA, then train + evaluate all 3 models.

set -e  # stop immediately if any command fails

echo "============================================"
echo " Credit Card Fraud Detection - Setup & Run"
echo "============================================"

# 1. Create virtual environment (skip if it already exists)
if [ ! -d "venv" ]; then
    echo "-> Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
echo "-> Installing dependencies (this can take a minute)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. Check dataset exists
if [ ! -f "data/creditcard.csv" ]; then
    echo ""
    echo "!! data/creditcard.csv not found."
    echo "   Download it from https://www.kaggle.com/mlg-ulb/creditcardfraud"
    echo "   and place it at data/creditcard.csv, then re-run this script."
    echo "   (Without it, the pipeline will auto-fall back to synthetic demo data.)"
    echo ""
fi

# 5. Run EDA
echo "-> Running EDA (generates plots in outputs/)..."
python src/eda.py

# 6. Run full training + evaluation pipeline
echo "-> Training and evaluating all 3 models..."
python src/main.py

echo ""
echo "============================================"
echo " DONE. Check the outputs/ folder for:"
echo "   - eda_*.png            (exploratory plots)"
echo "   - precision_recall_curves.png"
echo "   - confusion_matrix_*.png"
echo "   - metrics.json         (final numbers)"
echo " Trained models are saved in models/*.joblib"
echo "============================================"
echo ""
echo "-> Launching browser dashboard (Ctrl+C to stop)..."
streamlit run app.py
