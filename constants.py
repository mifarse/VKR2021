"""Constants are described here."""
from pathlib import Path

JSON_DIR = "json"
HOME_DIR = Path("/home/urukov")  # Home directory.
SRC_DIR = HOME_DIR / "data" / "raw"  # Source directory.
BIN_DIR = HOME_DIR / "data" / "bin"  # Binary version of source datasets.
PREPARED_DIR = HOME_DIR / "data" / "prepared"  # Stores prepared datasets.
PROCESSED_DIR = HOME_DIR / "data" / "processed"  # Stores processed data.
CACHE_DIR = HOME_DIR / ".utils_cache"