"""Validate the full original PACO-Part or LVIS image pools and export fixed examples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
LOGS=ROOT.parent/"setup-logs"
DOWNLOADS=ROOT.parent/"downloads"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["paco","lvis"],required=True)
    args=parser.parse_args()
    name=args.dataset
    image_status=json.loads((LOGS/(name+"-images-status.json")).read_text())
    assert image_status["status"]=="completed"
    requirements_path=DOWNLOADS/(name+"-required-official-images.json")
    assert digest(requirements_path)==image_status["requirements_sha256"]
    requirements=json.loads(requirements_path.read_text())
    cache=json.loads((LOGS/"coco2017-image-cache-manifest.json").read_text())
    metadata=json.loads((LOGS/(name+"-metadata-check.json")).read_text())
    image_records={}
    for index,filename in enumerate(requirements["images"]):
        assert filename in cache
        path=ROOT/"data/LVIS/coco"/filename
        record=cache[filename]
        assert path.stat().st_size==record["bytes"] and digest(path)==record["sha256"],filename
        with Image.open(path) as image:
            assert image.format=="JPEG"
            image.verify()
        image_records[filename]=record
        if (index+1)%2000==0:
            print("IMAGE_CHECK",index+1,"of",len(requirements["images"]),flush=True)
    if name=="paco":
        from datasets.paco_part import DatasetPACOPart as DatasetClass
        counts=[79,82,66,76]
        tasks=2500
        metadata_files=[ROOT/"data/PACO-Part/paco"/n for n in ["paco_part_train.pkl","paco_part_val.pkl"]]
    else:
        from datasets.lvis import DatasetLVIS as DatasetClass
        counts=[92]*10
        tasks=2300
        metadata_files=[ROOT/"data/LVIS/lvis_val.pkl",ROOT/"data/LVIS/lvis_train.pkl"]
    groups=[]
    cases=[]
    categories=[]
    for fold,count in enumerate(counts):
        ds=DatasetClass(str(ROOT/"data"),fold=fold,shot=1)
        assert len(ds.class_ids)==count and len(ds)==tasks
        categories.extend(int(c) for c in ds.class_ids_ori)
        np.random.seed(0)
        batch=ds[0]
        label=int(batch["class_id"])
        example=ROOT/"data"/("PACO-Part" if name=="paco" else "LVIS")/"example_cases"/f"fold{fold}"
        example.mkdir(parents=True,exist_ok=True)
        batch["tgt_img"].save(example/"target.png")
        batch["ref_imgs"][0].save(example/"reference.png")
        for kind,mask in [("target",batch["tgt_mask"]),("reference",batch["ref_masks"][0])]:
            raw=mask.numpy()
            assert raw.ndim==2 and min(raw.shape)>0
            Image.fromarray((raw>0).astype(np.uint8)*255).save(example/(kind+"_mask.png"))
        cases.append({"fold":fold,"original_class_id":int(ds.class_ids_ori[label]),
                      "reference_image":str(example/"reference.png"),"reference_mask":str(example/"reference_mask.png"),
                      "target_image":str(example/"target.png"),"target_ground_truth_mask":str(example/"target_mask.png")})
        groups.append({"fold":fold,"class_count":count,"original_class_ids":[int(c) for c in ds.class_ids_ori],
                       "episodes":len(ds),"box_crop":getattr(ds,"box_crop",False)})
    assert len(categories)==len(set(categories))==metadata["classes"]
    snapshot=LOGS/(name+"-verified-image-manifest.json")
    snapshot.write_text(json.dumps(image_records,indent=2)+"\n")
    report={"status":"PASS","dataset":requirements["dataset"],"groups":groups,
            "classes":len(categories),"episodes":sum(g["episodes"] for g in groups),
            "required_images":len(image_records),"requirements_sha256":digest(requirements_path),
            "verified_image_manifest":str(snapshot),"verified_image_manifest_sha256":digest(snapshot),
            "metadata_sha256":{str(p.relative_to(ROOT)):digest(p) for p in metadata_files},
            "sampling":"Unmodified official loader; all referenced image pools checked; seed 0 first sample per fold exported.",
            "model_parameters":"Same frozen ViT-L/16, 1024, 1-shot, CRF640, SVD500, tau0.6, merge0.2. No tuning."}
    (LOGS/(name+"-data-check.json")).write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"examples"/(name+"-demo-cases.json")).write_text(json.dumps({"cases":cases},indent=2)+"\n")
    print("FULL_DATA_CHECK_PASS",name,len(categories),report["episodes"],len(image_records),flush=True)


if __name__=="__main__":
    main()
