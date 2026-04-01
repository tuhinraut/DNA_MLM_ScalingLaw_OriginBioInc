#!/usr/bin/env python3
"""Bulk download CDS sequences from NCBI Genomes FTP.

Downloads protein-coding CDS from NCBI without using API.
Uses FTP to get cds_from_genomic.fna.gz files from latest assemblies.

Usage:
    python download_ncbi_ftp.py
    python download_ncbi_ftp.py --species human mouse
"""

import argparse
import gzip
import urllib.request
from pathlib import Path
from html.parser import HTMLParser
import time


# Species with NCBI assembly accessions
SPECIES = [
    ("Human", "human", "GCF_000001405.40", "GRCh38.p14"),
    ("Mouse", "mouse", "GCF_000001635.27", "GRCm39"),
    ("Rat", "rat", "GCF_015227675.2", "mRatBN7.2"),
    ("Zebrafish", "zebrafish", "GCF_000002035.6", "GRCz11"),
    ("Chicken", "chicken", "GCF_016699485.2", "bGalGal1.mat.broiler.GRCg7b"),
    ("Cow", "cow", "GCF_002263795.3", "ARS-UCD1.3"),
    ("Pig", "pig", "GCF_000003025.6", "Sscrofa11.1"),
    ("Dog", "dog", "GCF_000002285.5", "Dog10K_Boxer_Tasha"),
    ("Horse", "horse", "GCF_002863925.1", "EquCab3.0"),
    ("Rhesus macaque", "macaque", "GCF_003339765.1", "Mmul_10"),
    ("Chimpanzee", "chimp", "GCF_000001515.7", "Pan_tro_3.0"),
    ("Gorilla", "gorilla", "GCF_008122165.1", "Kamilah_GGO_v0"),
    ("Orangutan", "orangutan", "GCF_000001545.3", "Susie_PAB_v2"),
    ("Cat", "cat", "GCF_000181335.3", "Felis_catus_9.0"),
    ("Rabbit", "rabbit", "GCF_000003625.3", "OryCun2.0"),
    ("Sheep", "sheep", "GCF_016772045.1", "ARS-UI_Ramb_v2.0"),
    ("Goat", "goat", "GCF_001704415.2", "ARS1"),
    ("Fruit fly", "fly", "GCF_000001215.4", "Release 6 plus ISO1 MT"),
    ("C. elegans", "worm", "GCF_000002985.6", "WBcel235"),
    ("Yeast", "yeast", "GCF_000146045.2", "R64"),
    ("Platypus", "platypus", "GCF_004115215.2", "mOrnAna1.p.v1"),
    ("Opossum", "opossum", "GCF_000002295.2", "monDom5"),
    ("Wallaby", "wallaby", "GCF_027887165.1", "mPetWal1.pri"),
    ("Tasmanian devil", "devil", "GCF_902635505.1", "mSarHar1.11"),
    ("Green anole", "lizard", "GCF_090700615.1", "AnoCar2.0v2"),
    ("Xenopus", "xenopus", "GCF_000004195.4", "Xenopus_tropicalis_v9.1"),
    ("Frog", "frog", "GCF_905171775.1", "Xenopus_laevis_v10.1"),
    ("Pufferfish", "pufferfish", "GCF_000180735.1", "TETRAODON8"),
    ("Medaka", "medaka", "GCF_002234675.1", "ASM223467v1"),
    ("Stickleback", "stickleback", "GCF_016920845.1", "ASM1692084v1"),
    ("Turbot", "turbot", "GCF_008922235.1", "ASM892223v1"),
    ("Salmon", "salmon", "GCF_905237065.1", "Ssal_v3.1"),
    ("Trout", "trout", "GCF_013265735.2", "Omyk_1.1"),
    ("Cod", "cod", "GCF_902167405.1", "gadMor3.0"),
    ("Shark", "shark", "GCF_902713615.1", "sAmbRad1.1"),
    ("Lamprey", "lamprey", "GCF_010993605.1", "kPetMar1.pri"),
    ("Lancelet", "lancelet", "GCF_000003815.1", "BraFlo1"),
    ("Sea urchin", "urchin", "GCF_000002235.2", "Spur_5.0"),
    ("Starfish", "starfish", "GCF_013883685.1", "echinoderm"),
]

OUTPUT_DIR = Path("ncbi_ftp_output")
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


class LinkParser(HTMLParser):
    """Parse HTML to find directory links."""
    def __init__(self):
        super().__init__()
        self.links = []
        self.current_href = None
        
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.current_href = value
                    
    def handle_data(self, data):
        if self.current_href and data.strip():
            self.links.append((self.current_href, data.strip()))
            self.current_href = None


def split_accession(accession):
    """Split GCF_XXXXXXXXX.X into path components."""
    prefix = accession[:3]  # GCF
    nums = accession[4:].split('.')[0]  # 000001405
    # Path: GCF/000/001/405/
    path_parts = [prefix] + [nums[i:i+3] for i in range(0, len(nums), 3)]
    return '/'.join(path_parts)


def find_assembly_url(accession, assembly_name):
    """Build URL to assembly directory."""
    acc_path = split_accession(accession)
    base_url = f"{BASE_URL}/{acc_path}"
    
    # Try to find the exact directory
    try:
        req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        parser = LinkParser()
        parser.feed(html)
        
        # Find directory matching accession
        for href, _ in parser.links:
            if href.startswith(accession):
                return f"{base_url}/{href}"
                
    except Exception:
        pass
    
    # Fallback to constructed URL
    return f"{base_url}/{accession}_{assembly_name.replace(' ', '_')}"


def find_cds_file(assembly_url):
    """Find the CDS file URL in assembly directory."""
    try:
        req = urllib.request.Request(assembly_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        parser = LinkParser()
        parser.feed(html)
        
        # Look for cds_from_genomic.fna.gz
        for href, _ in parser.links:
            if 'cds_from_genomic.fna.gz' in href:
                return f"{assembly_url}/{href}"
                
    except Exception:
        pass
    
    return None


def download_file(url, output_path, timeout=300):
    """Download file with progress indication."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; BioDownloader/1.0)'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024**2)
                        total_mb = total_size / (1024**2)
                        print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)", end='', flush=True)
        
        print()  # Newline after progress
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n    ✗ HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n    ✗ Error: {e}")
        return False


def process_cds_file(gz_path, output_path):
    """Extract and filter CDS sequences."""
    try:
        print(f"  Extracting sequences...", end='', flush=True)
        
        count = 0
        with gzip.open(gz_path, 'rt') as gz_in, open(output_path, 'w') as out:
            header = None
            seq_lines = []
            
            for line in gz_in:
                line = line.strip()
                if line.startswith('>'):
                    # Write previous sequence
                    if header and seq_lines:
                        out.write(header + '\n')
                        out.write('\n'.join(seq_lines) + '\n')
                        count += 1
                    header = line
                    seq_lines = []
                else:
                    seq_lines.append(line)
            
            # Write last sequence
            if header and seq_lines:
                out.write(header + '\n')
                out.write('\n'.join(seq_lines) + '\n')
                count += 1
        
        print(f" ✓")
        return count
        
    except Exception as e:
        print(f" ✗ Error: {e}")
        return 0


def download_species(common_name, label, accession, assembly_name):
    """Download CDS for one species."""
    print(f"\n{'='*60}")
    print(f"Downloading: {common_name} ({label})")
    print(f"Assembly: {accession} {assembly_name}")
    print(f"{'='*60}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{label}_CDS.fasta"
    
    if output_file.exists() and output_file.stat().st_size > 10000:
        size_mb = output_file.stat().st_size / (1024**2)
        print(f"  ✓ File exists ({size_mb:.1f} MB) - skipping")
        return True
    
    # Find assembly URL
    print(f"  Finding assembly...")
    assembly_url = find_assembly_url(accession, assembly_name)
    
    # Find CDS file
    cds_url = find_cds_file(assembly_url)
    if not cds_url:
        print(f"  ✗ Could not find CDS file")
        return False
    
    print(f"  Found: {cds_url.split('/')[-1]}")
    
    # Download
    temp_gz = OUTPUT_DIR / f"{label}_temp_cds.fna.gz"
    print(f"  Downloading...")
    
    if not download_file(cds_url, temp_gz):
        if temp_gz.exists():
            temp_gz.unlink()
        return False
    
    # Process
    count = process_cds_file(temp_gz, output_file)
    temp_gz.unlink()
    
    if count > 0:
        size_mb = output_file.stat().st_size / (1024**2)
        print(f"  CDS sequences: {count:,}")
        print(f"  Output size: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ No sequences extracted")
        return False


def main():
    parser = argparse.ArgumentParser(description="NCBI Genomes FTP CDS downloader")
    parser.add_argument("--species", nargs="+", help="Species labels to download")
    parser.add_argument("--limit", type=int, help="Download first N species")
    args = parser.parse_args()
    
    species_list = SPECIES
    if args.species:
        species_list = [s for s in SPECIES if s[1] in args.species]
    if args.limit:
        species_list = species_list[:args.limit]
    
    if not species_list:
        print(f"Available species: {[s[1] for s in SPECIES]}")
        return
    
    print(f"Downloading {len(species_list)} species from NCBI Genomes FTP")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Base URL: {BASE_URL}")
    
    success = 0
    for common, label, accession, assembly in species_list:
        if download_species(common, label, accession, assembly):
            success += 1
        time.sleep(0.5)  # Be polite to server
    
    print(f"\n{'='*60}")
    print(f"Results: {success}/{len(species_list)} species downloaded")
    print(f"{'='*60}")
    
    # Summary
    files = sorted(OUTPUT_DIR.glob("*_CDS.fasta"))
    if files:
        total_mb = sum(f.stat().st_size for f in files) / (1024**2)
        print(f"\nDownloaded {len(files)} files ({total_mb:.1f} MB total):")
        for f in files:
            mb = f.stat().st_size / (1024**2)
            print(f"  {f.name}: {mb:.1f} MB")


if __name__ == "__main__":
    main()
