"""Download the Inside Airbnb detailed listings for NYC, Paris and Berlin.

Usage:  python scripts/download_data.py [--force]

Inside Airbnb only keeps recent snapshots online. If a URL stops working,
open https://insideairbnb.com/get-the-data/, copy the newest
"listings.csv.gz" link for the city and paste it into src/config.py.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for city, info in cfg.CITIES.items():
        out = cfg.DATA_DIR / f"{city}_listings.csv.gz"
        if out.exists() and not args.force:
            print(f"[skip] {out.name} already exists")
            continue
        print(f"[download] {info['name']}: {info['url']}")
        urllib.request.urlretrieve(info["url"], out)
        print(f"           -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
