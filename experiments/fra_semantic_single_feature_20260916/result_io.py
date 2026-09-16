"""Read plain or losslessly compressed JSON results; write deterministic gzip."""
import gzip
import json
from pathlib import Path


def result_exists(path):
    path = Path(path)
    return path.exists() or Path(str(path) + '.gz').exists()


def read_json(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + '.gz')
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def write_json_gz(path, data):
    path = Path(path)
    if path.suffix != '.gz':
        path = Path(str(path) + '.gz')
    encoded = json.dumps(data, indent=2, allow_nan=False).encode('utf-8')
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    return path
