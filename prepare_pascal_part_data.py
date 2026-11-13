"""Prepare the original VOC2010 images and author-provided PASCAL-Part annotations."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT.parent / "downloads"
LOGS = ROOT.parent / "setup-logs"
ANNO_SHA = "e79ed0e85c4d732ece744b90d4b24f6d2ae60db40531303ac9dc284bfb35a1fd"
VOC_MD5 = "da459979d0c395079b5c75ee67908abb"


def digest(path, algorithm="sha256"):
    h=hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    image_report=json.loads((LOGS/"pascal-part-images-status.json").read_text())
    assert image_report["status"]=="completed" and image_report["md5"]==VOC_MD5
    images=DOWNLOADS/"VOCtrainval_03-May-2010.tar"
    annos=DOWNLOADS/"pascal-part-annotations.zip"
    assert digest(images,"md5")==VOC_MD5
    assert digest(annos)==ANNO_SHA
    final=ROOT/"data/Pascal-Part"
    assert not final.exists(), "Inspect existing Pascal-Part data before rerunning."
    stage_parent=ROOT/"data"/("pascal-part-staging-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    stage=stage_parent/"Pascal-Part"
    voc=stage/"VOCdevkit/VOC2010"
    (voc/"JPEGImages").mkdir(parents=True)
    anno_dir=voc/"Annotations_Part_json_merged_part_classes"
    anno_dir.mkdir()
    image_count=0
    with tarfile.open(images) as archive:
        for member in archive:
            parts=Path(member.name).parts
            if not member.isfile() or len(parts)!=4 or parts[:3]!=("VOCdevkit","VOC2010","JPEGImages") or not member.name.endswith(".jpg"):
                continue
            target=voc/"JPEGImages"/parts[-1]
            assert not target.exists()
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source,output)
            image_count+=1
    assert image_count==11321,image_count
    annotation_count=0
    with zipfile.ZipFile(annos) as archive:
        assert archive.testzip() is None
        for member in archive.infolist():
            parts=Path(member.filename).parts
            if member.is_dir():
                continue
            if parts==("pascal","all_obj_part_to_image.json"):
                (voc/parts[-1]).write_bytes(archive.read(member))
            elif len(parts)==3 and parts[:2]==("pascal","Annotations_Part_json_merged_part_classes") and parts[-1].endswith(".json"):
                (anno_dir/parts[-1]).write_bytes(archive.read(member))
                annotation_count+=1
    metadata=json.loads((voc/"all_obj_part_to_image.json").read_text())
    required=set()
    for category in metadata.values():
        for obj in category["object"].values():
            for part in obj["part"].values():
                required.update(part["val"])
    for name in sorted(required):
        assert Path(name).name==name
        image=voc/"JPEGImages"/(name+".jpg")
        annotation=anno_dir/(name+".json")
        assert image.is_file() and annotation.is_file(),name
        with Image.open(image) as sample:
            w,h=sample.size
        data=json.loads(annotation.read_text())
        for obj in data["object"]:
            for part in obj["parts"]:
                assert all(mask["size"]==[h,w] for mask in part["mask"]),name
    from datasets.pascal_part import DatasetPASCALPart
    groups=[]
    cases=[]
    for fold in range(4):
        ds=DatasetPASCALPart(str(stage_parent),fold=fold,shot=1)
        assert ds.box_crop is True
        assert len(ds.class_ids)==[31,4,5,16][fold]
        assert len(ds)==[2500,959,2500,2500][fold]
        for name in ds.cat_part_name:
            obj,part=name.split("+")
            assert len(set(ds.cat_annos["object"][obj]["part"][part]["val"]))>=2
        np.random.seed(0)
        batch=ds[0]
        assert int(batch["class_id"])==0
        example=stage/"example_cases"/f"fold{fold}"
        example.mkdir(parents=True)
        batch["tgt_img"].save(example/"target.png")
        batch["ref_imgs"][0].save(example/"reference.png")
        for kind,mask in [("target",batch["tgt_mask"]),("reference",batch["ref_masks"][0])]:
            raw=mask.numpy()
            assert raw.ndim==2 and min(raw.shape)>0
            Image.fromarray((raw>0).astype(np.uint8)*255).save(example/(kind+"_mask.png"))
        cases.append({"fold":fold,"semantic_group":ds.cat,"part_name":ds.cat_part_name[0],
                      "reference_image":str(final/"example_cases"/f"fold{fold}/reference.png"),
                      "reference_mask":str(final/"example_cases"/f"fold{fold}/reference_mask.png"),
                      "target_image":str(final/"example_cases"/f"fold{fold}/target.png"),
                      "target_ground_truth_mask":str(final/"example_cases"/f"fold{fold}/target_mask.png")})
        groups.append({"fold":fold,"semantic_group":ds.cat,"class_ids":ds.class_ids,
                       "part_names":ds.cat_part_name,"class_count":len(ds.class_ids),"episodes":len(ds),
                       "box_crop":ds.box_crop})
    stage.rename(final)
    report={"status":"PASS","images":image_count,"annotations":annotation_count,
            "required_validation_image_ids":len(required),"groups":groups,"classes":56,
            "episodes":sum(g["episodes"] for g in groups),"box_crop":True,
            "archives":{"images":{"file":images.name,"md5":VOC_MD5,"sha256":digest(images)},
                        "annotations":{"file":annos.name,"sha256":ANNO_SHA,"zip_crc":"PASS"}},
            "sampling":"Unmodified DatasetPASCALPart; reference and target sampled from official val pools; default object-box crop preserved.",
            "aggregation_note":"Group class counts differ. Preserve per-class IoUs and report class macro and group macro separately; released docs do not explicitly specify cross-group aggregation."}
    (LOGS/"pascal-part-data-check.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"examples/pascal-part-demo-cases.json").write_text(json.dumps({"selection":"seed=0, first official sample per group","cases":cases},indent=2)+"\n")
    print("PASCAL_PART_DATA_PASS",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
