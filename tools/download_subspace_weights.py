"""Download the pinned public HF Giant model with verified resumable ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

REPO = 'facebook/dinov2-with-registers-giant'
ROOT = Path(__file__).resolve().parents[1]/'weights/dinov2_with_registers_giant'
PROXIES = {'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}


def get(url, **kwargs):
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=PROXIES, timeout=(30, 90), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2*(attempt+1))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    info = get(f'https://huggingface.co/api/models/{REPO}?blobs=true').json()
    sha = info['sha']
    name = 'model.safetensors'
    entry = next(x for x in info['siblings'] if x['rfilename']==name)
    size, digest = entry['size'], entry['lfs']['sha256']
    for small in ['config.json','preprocessor_config.json','README.md']:
        (ROOT/small).write_bytes(get(f'https://huggingface.co/{REPO}/resolve/{sha}/{small}').content)
    target = ROOT/name
    if target.exists() and hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()==digest:
        print('Weights already verified', flush=True)
        return
    partial = ROOT/(name+'.part')
    progress = ROOT/'download-progress.json'
    chunk = 32*1024*1024
    done = set()
    if progress.exists() and partial.exists():
        saved = json.loads(progress.read_text())
        if saved['revision']==sha and partial.stat().st_size==size:
            done = set(saved['done'])
    if not done:
        with partial.open('wb') as file:
            file.truncate(size)
    count = (size+chunk-1)//chunk

    def fetch(index):
        start, end = index*chunk, min(size,(index+1)*chunk)-1
        for attempt in range(5):
            try:
                url = f'https://huggingface.co/{REPO}/resolve/{sha}/{name}?download=true&chunk={index}'
                with get(url, headers={'Range':f'bytes={start}-{end}'}, stream=True) as response:
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError(f'Unexpected range response: {response.status_code} {response.headers.get("Content-Range")}')
                    written = 0
                    with partial.open('r+b') as file:
                        file.seek(start)
                        for block in response.iter_content(1024*1024):
                            file.write(block)
                            written += len(block)
                    if written != end-start+1:
                        raise ValueError('Incomplete range')
                return index
            except (requests.RequestException, ValueError):
                if attempt==4:
                    raise
                time.sleep(2*(attempt+1))

    print(f'Download {size/1e9:.2f} GB, {len(done)}/{count} ranges ready, revision {sha}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch,i) for i in range(count) if i not in done]
        for future in concurrent.futures.as_completed(futures):
            done.add(future.result())
            progress.write_text(json.dumps({'revision':sha,'done':sorted(done)}))
            print(f'Completed {len(done)}/{count} ranges ({min(len(done)*chunk,size)/1e9:.2f} GB)',flush=True)
    with partial.open('rb') as file:
        actual = hashlib.file_digest(file, 'sha256').hexdigest()
    if actual != digest:
        raise ValueError('Downloaded weights failed SHA256 verification')
    partial.replace(target)
    (ROOT/'source.json').write_text(json.dumps({'repo':REPO,'revision':sha,'sha256':digest,'remote_sha256_verified':True},indent=2))
    print('Weights complete; SHA256 verified',flush=True)


if __name__=='__main__':
    main()
