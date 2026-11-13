"""Acquire every official image referenced by a complete benchmark's metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
from PIL import Image

ROOT=Path(__file__).resolve().parent
LOGS=ROOT.parent/"setup-logs"
DEST=ROOT/"data/LVIS/coco"
OLD=ROOT/"data/COCO2014/val2014"
CACHE_MANIFEST=LOGS/"coco2017-image-cache-manifest.json"
BASE_URL="https://s3.amazonaws.com/images.cocodataset.org/"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements",type=Path,required=True)
    parser.add_argument("--name",required=True,choices=["paco","lvis"])
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    requirements=json.loads(args.requirements.read_text())
    names=sorted(set(requirements["images"]))
    assert all(re.fullmatch(r"(?:train2017|val2017)/[0-9]{12}\.jpg",n) for n in names)
    records=json.loads(CACHE_MANIFEST.read_text()) if CACHE_MANIFEST.exists() else {}
    status=LOGS/(args.name+"-images-status.json")
    lock=threading.Lock()
    cancel=threading.Event()
    state={"status":"preparing_referenced_images","dataset":requirements["dataset"],"pid":os.getpid(),
           "workers":args.workers,"required_images":len(names),"verified_images":0,"reused_images":0,"downloaded_images":0,"downloaded_bytes":0,
           "source":BASE_URL,"requirements_sha256":hashlib.sha256(args.requirements.read_bytes()).hexdigest()}
    def save():
        temporary=status.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state,indent=2)+"\n")
        temporary.replace(status)
        temporary=CACHE_MANIFEST.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records,indent=2)+"\n")
        temporary.replace(CACHE_MANIFEST)
    save()
    def fetch(name):
        destination=DEST/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        url=BASE_URL+name
        reused=False
        downloaded=0
        if destination.exists() and name in records:
            data=destination.read_bytes()
            assert hashlib.sha256(data).hexdigest()==records[name]["sha256"],name
            return name,records[name],False,0
        for attempt in range(3):
            if cancel.is_set():
                raise RuntimeError("Cancelled after an image failed.")
            try:
                candidate=OLD/("COCO_val2014_"+Path(name).name)
                if candidate.exists():
                    data=candidate.read_bytes()
                    with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=30) as response:
                        assert response.status==200
                        etag=response.headers.get("ETag","").strip('"')
                        expected_length=int(response.headers["Content-Length"])
                    if re.fullmatch("[0-9a-f]{32}",etag) and hashlib.md5(data).hexdigest()==etag and len(data)==expected_length:
                        if destination.exists():
                            assert destination.read_bytes()==data,name
                        else:
                            os.link(candidate,destination)
                        reused=True
                if not reused:
                    with urllib.request.urlopen(url,timeout=45) as response:
                        assert response.status==200
                        data=response.read()
                        expected_length=int(response.headers["Content-Length"])
                        etag=response.headers.get("ETag","").strip('"')
                    assert len(data)==expected_length
                    if re.fullmatch("[0-9a-f]{32}",etag):
                        assert hashlib.md5(data).hexdigest()==etag,name
                    with Image.open(io.BytesIO(data)) as image:
                        assert image.format=="JPEG"
                        image.verify()
                    temporary=destination.with_suffix(".jpg.part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                    downloaded=len(data)
                record={"url":url,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                        "md5":hashlib.md5(data).hexdigest(),"official_object_etag":etag,
                        "acquisition":"verified_existing_COCO2014_image" if reused else "official_COCO2017_object"}
                return name,record,reused,downloaded
            except Exception:
                if attempt==2:
                    raise
                time.sleep(2)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(fetch,n) for n in names]
            try:
                for future in as_completed(futures):
                    name,record,reused,downloaded=future.result()
                    with lock:
                        records[name]=record
                        state["verified_images"]+=1
                        state["reused_images"]+=int(reused)
                        state["downloaded_images"]+=int(downloaded>0)
                        state["downloaded_bytes"]+=downloaded
                        if state["verified_images"]%20==0:
                            save()
                            print("VERIFIED_IMAGES",state["verified_images"],"of",len(names),flush=True)
            except Exception:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
        assert state["verified_images"]==len(names)
        state["status"]="completed"
        state["note"]="All official-loader referenced image pools are complete; full COCO2017 train/val archives were not required."
        save()
        print("REFERENCED_COCO2017_IMAGES_PASS",json.dumps(state),flush=True)
    except Exception as error:
        state.update(status="failed",error=str(error))
        save()
        raise


if __name__=="__main__":
    with (LOGS/"coco2017-image-cache.lock").open("a") as cache_lock:
        fcntl.flock(cache_lock,fcntl.LOCK_EX)
        main()
