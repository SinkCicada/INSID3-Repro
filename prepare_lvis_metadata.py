"""Inspect the original LVIS metadata and enumerate every image required by all ten folds."""
import collections
import hashlib
import json
from pathlib import Path
import pickle
import re
import zipfile

ROOT=Path(__file__).resolve().parent
DOWNLOADS=ROOT.parent/"downloads"
LOGS=ROOT.parent/"setup-logs"


def main():
    record=json.loads((DOWNLOADS/"lvis-local-download-check.json").read_text())[0]
    source=DOWNLOADS/"lvis.zip"
    assert record["file"]==source.name and record["zip_crc"]=="PASS"
    sha=hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            sha.update(block)
    assert source.stat().st_size==record["bytes"] and sha.hexdigest()==record["sha256"]
    dest=ROOT/"data/LVIS"
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
        for basename in ["lvis_train.pkl","lvis_val.pkl"]:
            entries=[n for n in archive.namelist() if Path(n).name==basename]
            assert len(entries)==1,entries
            target=dest/basename
            assert not target.exists(), "Inspect already prepared LVIS metadata before rerunning."
            target.write_bytes(archive.read(entries[0]))
    from datasets.lvis import DatasetLVIS
    groups=[]
    selected=set()
    for fold in range(10):
        ds=DatasetLVIS(str(ROOT/"data"),fold=fold,shot=1)
        assert len(ds.class_ids)==92 and len(ds)==2300
        selected.update(int(c) for c in ds.class_ids_ori)
        groups.append({"fold":fold,"class_count":len(ds.class_ids),"episodes":len(ds),
                       "original_class_ids":[int(c) for c in ds.class_ids_ori]})
    assert len(selected)==920
    with (dest/"lvis_val.pkl").open("rb") as stream:
        data=pickle.load(stream)
    images=sorted({str(n) for category in selected for n in data[category]})
    assert all(re.fullmatch(r"(?:train2017|val2017)/[0-9]{12}\.jpg",name) for name in images)
    requirements={"dataset":"LVIS-92i","images":images,"source":"all selected class pools from official lvis_val.pkl",
                  "annotations_sha256":record["sha256"],"classes":920,"folds":10}
    path=DOWNLOADS/"lvis-required-official-images.json"
    path.write_text(json.dumps(requirements,indent=2)+"\n")
    old=ROOT/"data/COCO2014/val2014"
    candidates=sum((old/("COCO_val2014_"+Path(name).name)).is_file() for name in images)
    report={"status":"METADATA_PASS_IMAGES_PENDING","groups":groups,"classes":920,"episodes":23000,
            "required_images":len(images),"image_subsets":dict(collections.Counter(n.split("/")[0] for n in images)),
            "existing_COCO2014_candidates":candidates,"annotations_sha256":record["sha256"],"requirements":str(path)}
    (LOGS/"lvis-metadata-check.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LVIS_METADATA_PASS",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
