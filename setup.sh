#!/bin/bash
# One-time environment setup: installs dependencies and downloads CDS data.
set -euo pipefail

echo "[setup] checking Python..."
python3 --version

echo "[setup] installing dependencies..."
python3 -m pip install -r requirements.txt

echo "[setup] making shell/python scripts executable..."
chmod +x scripts/*.sh data_downloaders/*.py 2>/dev/null || true

# --- data download -----------------------------------------------------------
FASTA_COUNT=$(ls ncbi_ftp_output/*_CDS.fasta 2>/dev/null | wc -l)
if [[ "$FASTA_COUNT" -ge 33 ]]; then
  echo "[setup] data already present: $FASTA_COUNT species in ncbi_ftp_output/"
else
  echo "[setup] downloading CDS data for all species (~4.5 GB, may take 30-60 min)..."
  echo "[setup] to download a subset only, Ctrl-C and run:"
  echo "        python data_downloaders/download_ncbi_ftp.py --species human mouse rat"
  echo ""
  python3 data_downloaders/download_ncbi_ftp.py
fi

cat <<'EOF'

Next steps:
  quick sanity check:   ./scripts/test_scaling_orchestration.sh
  full sweep:           ./scripts/run_complete_scaling_analysis.sh
EOF
