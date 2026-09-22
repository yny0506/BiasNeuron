
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

COMMIT = '9eba1ba179cc3169f75a1c03a8b392a58d188303'
BASE_URL = 'https://raw.githubusercontent.com/nyu-mll/BBQ/{commit}/data/{name}.jsonl'
CHECKSUMS = {
    'SES': '9f92754bb037b0982604b9112705fb81d60a19d9e759c67e4a85e484a070f528',
    'Age': '46e805b3fc2d8cbd26eeb8e8430d98cf7b2dc9c83574ff3674e8ce4f0fca2a60',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/BBQ')
    parser.add_argument('--commit', default=COMMIT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, expected in CHECKSUMS.items():
        path = out_dir / f'{name}.jsonl'
        if not path.exists() or sha256(path) != expected:
            url = BASE_URL.format(commit=args.commit, name=name)
            print(f'downloading {url}')
            partial = path.with_suffix('.jsonl.part')
            urllib.request.urlretrieve(url, partial)
            if sha256(partial) == expected:
                os.replace(partial, path)
            else:
                print(f'{path}  {sha256(partial)}  MISMATCH (expected {expected})')
                partial.unlink()
                failed.append(name)
                continue
        print(f'{path}  {sha256(path)}  ok')

    if failed:
        sys.exit(f'checksum mismatch for {", ".join(failed)}; refusing to continue')


if __name__ == '__main__':
    main()
