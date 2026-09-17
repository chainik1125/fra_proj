"""Repack result JSON without losing measurements; keep every blob below 1 MB."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from analyze import load


def pack(stem):
    stem = Path(stem)
    result = load(stem)
    raw = json.dumps(result, allow_nan=False)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    single = gzip.compress(raw.encode(), mtime=0)
    outputs = {}
    if len(single) < 950000:
        outputs[stem.with_name(stem.name + '.json.gz')] = single
    else:
        # JSON wrappers add only a small overhead. Subdivide unusually dense
        # fragments until the actual compressed wrapper fits the hook limit.
        fragments = [raw[i:i+4000000] for i in range(0, len(raw), 4000000)]
        while True:
            blobs = [gzip.compress(json.dumps({'part': i, 'parts': len(fragments),
                      'json_fragment': chunk}).encode(), mtime=0)
                     for i, chunk in enumerate(fragments)]
            if max(map(len, blobs)) < 950000:
                break
            revised = []
            for chunk, blob in zip(fragments, blobs):
                if len(blob) >= 950000:
                    mid = len(chunk)//2
                    revised.extend([chunk[:mid], chunk[mid:]])
                else:
                    revised.append(chunk)
            fragments = revised
        outputs = {stem.with_name(stem.name + f'.part{i:03}.json.gz'): b
                   for i, b in enumerate(blobs)}
    old = set(stem.parent.glob(stem.name + '.part*.json.gz'))
    old.add(stem.with_name(stem.name + '.json.gz'))
    for path, blob in outputs.items():
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(blob)
        tmp.replace(path)
    for path in old - outputs.keys():
        path.unlink(missing_ok=True)
    check = json.dumps(load(stem), allow_nan=False)
    assert hashlib.sha256(check.encode()).hexdigest() == digest
    print(stem.name, 'files', len(outputs), 'largest', max(map(len, outputs.values())),
          'JSON SHA256', digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stems', nargs='+')
    for stem in parser.parse_args().stems:
        pack(stem)
