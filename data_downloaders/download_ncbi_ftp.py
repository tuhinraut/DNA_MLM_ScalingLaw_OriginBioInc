#!/usr/bin/env python3
"""Download protein-coding CDS FASTA from NCBI Genomes FTP for a list of species.

Usage:
    python data_downloaders/download_ncbi_ftp.py              # all species
    python data_downloaders/download_ncbi_ftp.py --species human mouse
    python data_downloaders/download_ncbi_ftp.py --limit 3    # first 3 species
"""

import argparse
import gzip
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from tqdm import tqdm


SPECIES = [
    ("Human",            "human",       "GCF_000001405.40", "GRCh38.p14"),
    ("Mouse",            "mouse",       "GCF_000001635.27", "GRCm39"),
    ("Rat",              "rat",         "GCF_015227675.2",  "mRatBN7.2"),
    ("Zebrafish",        "zebrafish",   "GCF_000002035.6",  "GRCz11"),
    ("Chicken",          "chicken",     "GCF_016699485.2",  "bGalGal1.mat.broiler.GRCg7b"),
    ("Cow",              "cow",         "GCF_002263795.3",  "ARS-UCD1.3"),
    ("Pig",              "pig",         "GCF_000003025.6",  "Sscrofa11.1"),
    ("Dog",              "dog",         "GCF_000002285.5",  "Dog10K_Boxer_Tasha"),
    ("Horse",            "horse",       "GCF_002863925.1",  "EquCab3.0"),
    ("Rhesus macaque",   "macaque",     "GCF_003339765.1",  "Mmul_10"),
    ("Chimpanzee",       "chimp",       "GCF_000001515.7",  "Pan_tro_3.0"),
    ("Gorilla",          "gorilla",     "GCF_008122165.1",  "Kamilah_GGO_v0"),
    ("Orangutan",        "orangutan",   "GCF_000001545.3",  "Susie_PAB_v2"),
    ("Cat",              "cat",         "GCF_000181335.3",  "Felis_catus_9.0"),
    ("Rabbit",           "rabbit",      "GCF_000003625.3",  "OryCun2.0"),
    ("Sheep",            "sheep",       "GCF_016772045.1",  "ARS-UI_Ramb_v2.0"),
    ("Goat",             "goat",        "GCF_001704415.2",  "ARS1"),
    ("Fruit fly",        "fly",         "GCF_000001215.4",  "Release 6 plus ISO1 MT"),
    ("C. elegans",       "worm",        "GCF_000002985.6",  "WBcel235"),
    ("Yeast",            "yeast",       "GCF_000146045.2",  "R64"),
    ("Platypus",         "platypus",    "GCF_004115215.2",  "mOrnAna1.p.v1"),
    ("Opossum",          "opossum",     "GCF_000002295.2",  "monDom5"),
    ("Wallaby",          "wallaby",     "GCF_027887165.1",  "mPetWal1.pri"),
    ("Tasmanian devil",  "devil",       "GCF_902635505.1",  "mSarHar1.11"),
    ("Green anole",      "lizard",      "GCF_090700615.1",  "AnoCar2.0v2"),
    ("Xenopus",          "xenopus",     "GCF_000004195.4",  "Xenopus_tropicalis_v9.1"),
    ("Frog",             "frog",        "GCF_905171775.1",  "Xenopus_laevis_v10.1"),
    ("Pufferfish",       "pufferfish",  "GCF_000180735.1",  "TETRAODON8"),
    ("Medaka",           "medaka",      "GCF_002234675.1",  "ASM223467v1"),
    ("Stickleback",      "stickleback", "GCF_016920845.1",  "ASM1692084v1"),
    ("Turbot",           "turbot",      "GCF_008922235.1",  "ASM892223v1"),
    ("Salmon",           "salmon",      "GCF_905237065.1",  "Ssal_v3.1"),
    ("Trout",            "trout",       "GCF_013265735.2",  "Omyk_1.1"),
    ("Cod",              "cod",         "GCF_902167405.1",  "gadMor3.0"),
    ("Shark",            "shark",       "GCF_902713615.1",  "sAmbRad1.1"),
    ("Lamprey",          "lamprey",     "GCF_010993605.1",  "kPetMar1.pri"),
    ("Lancelet",         "lancelet",    "GCF_000003815.1",  "BraFlo1"),
    ("Sea urchin",       "urchin",      "GCF_000002235.2",  "Spur_5.0"),
    ("Starfish",         "starfish",    "GCF_013883685.1",  "echinoderm"),
]

OUTPUT_DIR = Path("ncbi_ftp_output")
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
UA = {"User-Agent": "Mozilla/5.0 (compatible; BioDownloader/1.0)"}


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href":
                    self.hrefs.append(v)


def _list_dir(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8")
    p = _LinkParser(); p.feed(html)
    return p.hrefs


def _assembly_url(accession):
    # Path: GCF/000/001/405/ + matching directory.
    prefix = accession[:3]
    nums = accession[4:].split(".")[0]
    path = "/".join([prefix] + [nums[i:i + 3] for i in range(0, len(nums), 3)])
    base = f"{BASE_URL}/{path}"
    try:
        for href in _list_dir(base):
            if href.startswith(accession):
                return f"{base}/{href.rstrip('/')}"
    except Exception:
        pass
    return None


def _find_cds_url(assembly_url):
    try:
        for href in _list_dir(assembly_url):
            if "cds_from_genomic.fna.gz" in href:
                return f"{assembly_url}/{href}"
    except Exception:
        pass
    return None


def _download(url, out_path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True,
                    desc=f"  {out_path.name}", leave=False)
        with open(out_path, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def _extract_fasta(gz_path, out_path):
    count = 0
    with gzip.open(gz_path, "rt") as gz, open(out_path, "w") as out:
        for line in gz:
            if line.startswith(">"):
                count += 1
            out.write(line)
    return count


def download_species(common, label, accession, assembly_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fasta = OUTPUT_DIR / f"{label}_CDS.fasta"
    if out_fasta.exists() and out_fasta.stat().st_size > 10_000:
        return True, out_fasta.stat().st_size, None

    asm_url = _assembly_url(accession)
    if not asm_url:
        return False, 0, "assembly URL not found"
    cds_url = _find_cds_url(asm_url)
    if not cds_url:
        return False, 0, "CDS file not found"

    tmp = OUTPUT_DIR / f"{label}_temp_cds.fna.gz"
    try:
        _download(cds_url, tmp)
        _extract_fasta(tmp, out_fasta)
    except Exception as e:
        return False, 0, str(e)
    finally:
        if tmp.exists():
            tmp.unlink()
    return True, out_fasta.stat().st_size, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", help="labels to download (e.g. human mouse)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    species = SPECIES
    if args.species:
        species = [s for s in SPECIES if s[1] in args.species]
    if args.limit:
        species = species[:args.limit]
    if not species:
        print("available:", ", ".join(s[1] for s in SPECIES))
        return

    print(f"downloading {len(species)} species -> {OUTPUT_DIR}/")
    ok = 0
    for common, label, acc, asm in tqdm(species, desc="species", unit="sp"):
        success, size, err = download_species(common, label, acc, asm)
        if success:
            tqdm.write(f"  ok  {label:<12} {size / 1e6:7.1f} MB")
            ok += 1
        else:
            tqdm.write(f"  err {label:<12} ({err})")
        time.sleep(0.3)
    print(f"\n{ok}/{len(species)} species downloaded")


if __name__ == "__main__":
    main()
