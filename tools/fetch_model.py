"""Download immutable model inputs. Runtime assets never depend on this cache."""
from pathlib import Path
import hashlib
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = 'cfd4ba3b0ea411d40f404d226022bc70f9e06139'
FILES = {
 'config.json': ('config.json','c902d3a31e16ea08252f617a1a30550d957e597b3e6c1295cef8d4418121cd70'),
 'last.pt': ('checkpoints/last.pt','eb0d80dbd57a517473dcb8825f314040ef1e21bd8ec13e0af54cd7f6201e5f55'),
}
SOURCES = ('https://huggingface.co', 'https://hf-mirror.com')


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': 'samuel-live-asset-builder/1'})
    with urllib.request.urlopen(request, timeout=120) as source, destination.open('wb') as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def fetch():
    cache = ROOT / 'tools/cache'
    cache.mkdir(exist_ok=True)
    for name,(remote,digest) in FILES.items():
        path=cache/name
        if not path.exists():
            tmp=path.with_suffix('.part')
            failures=[]
            for base in SOURCES:
                try:
                    download(f'{base}/vvolhejn/samuel/resolve/{REV}/{remote}', tmp)
                    if hashlib.sha256(tmp.read_bytes()).hexdigest()!=digest:
                        raise ValueError(f'{name} checksum mismatch from {base}')
                    tmp.replace(path)
                    break
                except Exception as error:
                    failures.append(f'{base}: {error}')
                    tmp.unlink(missing_ok=True)
            else:
                raise RuntimeError(f'could not download {name}; ' + '; '.join(failures))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'{name} checksum mismatch')
        print(f'{name}: verified {digest}')
if __name__=='__main__': fetch()
