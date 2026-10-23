"""Fetch or verify the exact public source bytes used in this release."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    rows=json.loads((ROOT/'data/source_manifest.json').read_text())
    for row in rows:
        target=ROOT/row['path']
        if target.exists():
            if digest(target)!=row['sha256']:raise ValueError(f"Source changed: {target}")
            continue
        if args.check:raise FileNotFoundError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            downloaded=Path(tmp)/'download'
            with urllib.request.urlopen(row['url'],timeout=120) as src,downloaded.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            candidate=downloaded
            if row.get('zip_member'):
                if digest(downloaded)!=row['archive_sha256']:raise ValueError('Upstream archive changed')
                candidate=Path(tmp)/'extracted'
                with zipfile.ZipFile(downloaded) as z,z.open(row['zip_member']) as src,candidate.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
            if digest(candidate)!=row['sha256']:raise ValueError(f"Upstream source changed: {row['url']}")
            shutil.copyfile(candidate,target)
    print(f"Verified {len(rows)} required source files")
if __name__=='__main__':main()
