#!/bin/bash
# One-time setup for DNA MLM Scaling Package

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  DNA MLM Scaling Laws - Setup                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found"
    exit 1
fi

echo "✓ pip found"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install torch numpy biopython 2>/dev/null || pip3 install torch numpy biopython

echo "✓ Dependencies installed"

# Create experiment directories
echo ""
echo "Creating experiment directories..."
mkdir -p experiments/{logs,checkpoints,results,data_samples}
mkdir -p ncbi_ftp_output

echo "✓ Directories created"

# Make scripts executable
echo ""
echo "Setting permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x utils/*.py 2>/dev/null || true
chmod +x data_downloaders/*.py 2>/dev/null || true

echo "✓ Scripts made executable"

# Check for data
echo ""
echo "Checking for data..."
if [ -d "ncbi_ftp_output" ] && [ "$(ls -A ncbi_ftp_output 2>/dev/null)" ]; then
    echo "✓ NCBI data found: $(ls ncbi_ftp_output/*_CDS.fasta 2>/dev/null | wc -l) files"
else
    echo "⚠ No data found in ncbi_ftp_output/"
    echo "  Download data with: python data_downloaders/download_ncbi_ftp.py"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Download data (if not present):"
echo "     python data_downloaders/download_ncbi_ftp.py"
echo ""
echo "  2. Run complete analysis:"
echo "     ./scripts/run_complete_scaling_analysis.sh"
echo ""
echo "  3. Or run with cleanup:"
echo "     ./scripts/run_complete_scaling_analysis.sh --cleanup"
echo ""
