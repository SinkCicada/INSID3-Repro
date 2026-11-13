"""Run the author PerMIS generator with original masks and on-demand original ZIP frames."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import zipfile
import zlib
import numpy as np
from PIL import Image
from permis_remote_zip import RemoteZipReader

ROOT=Path(__file__).resolve().parent
LOGS=ROOT.parent/"setup-logs"
DOWNLOADS=ROOT.parent/"downloads"
SOURCE=ROOT/"deps/PDM"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    args=parser.parse_args()
    assert digest(args.annotations_zip)==args.expected_sha256,"Uploaded annotation ZIP SHA-256 mismatch"
    final=ROOT/"data/PerMIRS"
    assert not final.exists(),"Inspect an existing PerMIRS dataset before preparing a new one."
    index_status=json.loads((LOGS/"permis-zip-index-status.json").read_text())
    assert index_status["status"]=="completed"
    assert json.loads((LOGS/"permis-frame-access-check.json").read_text())["status"]=="PASS"
    source_manifest=json.loads((SOURCE/"SOURCE_MANIFEST.json").read_text())
    for record in source_manifest:
        assert digest(SOURCE/record["file"])==record["sha256"],record["file"]
    annotation=DOWNLOADS/"burst-test-all_classes.json"
    with zipfile.ZipFile(args.annotations_zip) as archive:
        members=[x for x in archive.infolist() if not x.is_dir() and Path(x.filename).parts[-2:]==("test","all_classes.json")]
        assert len(members)==1
        annotation.write_bytes(archive.read(members[0]))
    source_dir=SOURCE/"PerMIRS"
    sys.path[:0]=[str(source_dir),str(SOURCE)]
    import dataset as pdm_dataset
    import video as pdm_video
    raw_root=DOWNLOADS/"permis-required-frames"
    raw_root.mkdir(exist_ok=True)
    preview=pdm_dataset.BURSTDataset(str(annotation),images_base_dir=None)
    assert preview._split=="test"
    for item in preview._videos:
        path=(raw_root/preview._split/item["dataset"]/item["seq_name"]).resolve()
        assert raw_root.resolve() in path.parents
        path.mkdir(parents=True,exist_ok=True)
    video_count=preview.num_videos
    del preview
    gc.collect()
    stage_parent=ROOT/"data"/("permis-staging-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    stage=stage_parent/"PerMIRS"
    stage.mkdir(parents=True)
    original=source_dir/"permirs_gen_dataset.py"
    text=original.read_text()
    old='base_path = f"/PerMIRS/{i}"'
    new='base_path = os.path.join(args.save_dir, str(i))'
    assert text.count(old)==1
    patched=source_dir/"permirs_gen_dataset_local.py"
    patched.write_text(text.replace(old,new))
    (LOGS/"permis-generator-path.patch").write_text("- "+old+"\n+ "+new+"\n")
    spec=importlib.util.spec_from_file_location("insid3_pdm_permis_generator",patched)
    generator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    expected={}
    frame_records={}
    failures=[]
    original_getitem=pdm_dataset.BURSTDataset.__getitem__
    def attach_index(self,index):
        sample=original_getitem(self,index)
        sample._insid3_case_index=index
        return sample
    pdm_dataset.BURSTDataset.__getitem__=attach_index
    original_paths=pdm_video.BURSTVideo.images_paths
    with RemoteZipReader(LOGS/"permis-zip-index-status.json") as remote:
        with zipfile.ZipFile(remote) as archive:
            def ensure_paths(self,frame_indices=None):
                indices=list(range(self.num_annotated_frames)) if frame_indices is None else list(frame_indices)
                case=int(self._insid3_case_index)
                if len(indices)>=3:
                    assert len(indices)==3
                    expected[case]={"dataset":self.dataset,"video":self.name,"frame_indices":[int(i) for i in indices],"frames":[]}
                paths=[]
                try:
                    for index in indices:
                        path=(Path(self._images_dir)/self.annotated_image_paths[index]).resolve()
                        assert raw_root.resolve() in path.parents
                        entry="frames/"+path.relative_to(raw_root).as_posix()
                        info=archive.getinfo(entry)
                        assert not info.is_dir() and path.suffix.lower()==".jpg"
                        data=path.read_bytes() if path.exists() else None
                        if data is None or len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:
                            data=archive.read(info)  # Original compressed member; zipfile verifies entry CRC.
                            temporary=path.with_suffix(".jpg.part")
                            temporary.write_bytes(data);temporary.replace(path)
                        with Image.open(path) as image:image.verify()
                        record={"archive_entry":entry,"bytes":len(data),"zip_crc":info.CRC,"sha256":hashlib.sha256(data).hexdigest()}
                        frame_records[entry]=record
                        paths.append(record)
                    if case in expected:expected[case]["frames"]=paths
                    manifest={"cases":expected,"frames":frame_records,"network_bytes":remote.network_bytes,"network_requests":remote.network_requests}
                    temporary=LOGS/"permis-frame-acquisition.json.tmp"
                    temporary.write_text(json.dumps(manifest,indent=2)+"\n");temporary.replace(temporary.with_suffix(""))
                    print("PERMIS_SOURCE_FRAMES",case,len(expected),len(frame_records),flush=True)
                except Exception as error:
                    if len(indices)>=3:failures.append({"case":case,"error":str(error)})
                    raise
                return original_paths(self,indices)
            pdm_video.BURSTVideo.images_paths=ensure_paths
            generator.main(argparse.Namespace(images_base_dir=str(raw_root),annotations_file=str(annotation),
                                             first_frame_annotations=False,seq=None,save_dir=str(stage)))
            assert not failures,failures
            actual={int(path.name) for path in stage.iterdir() if path.is_dir()}
            assert actual==set(expected), (sorted(actual),sorted(expected))
            assert len(actual)==216, f"Official INSID3 preparation expects 216 cases; obtained {len(actual)}"
            for case in sorted(actual):
                folder=stage/str(case)
                masks=np.load(folder/"masks.npz.npy",allow_pickle=True)
                assert len(masks)==3
                tracks=[]
                for index in range(3):
                    assert len(masks[index])==1
                    track,mask=next(iter(masks[index].items()));tracks.append(int(track))
                    with Image.open(folder/f"{index}.jpg") as image:
                        assert mask.shape==(image.height,image.width)
                    assert mask.ndim==2 and mask.any() and set(np.unique(mask).tolist())<={False,True,0,1}
                    assert digest(folder/f"{index}.jpg")==expected[case]["frames"][index]["sha256"]
                assert len(set(tracks))==1
            acquisition_stats={"network_bytes":remote.network_bytes,"network_requests":remote.network_requests}
    child_code=("from datasets.permis import DatasetPerMis; d=DatasetPerMis("+repr(str(stage))+",shot=1); "
                "assert len(d)==432 and d.class_ids==[0]; print('OFFICIAL_PERMIS_LOADER_PASS',len(d))")
    child=subprocess.run([sys.executable,"-c",child_code],cwd=ROOT,capture_output=True,text=True)
    (LOGS/"permis-loader-check.log").write_text(child.stdout+child.stderr)
    assert child.returncode==0,child.stderr[-2000:]
    stage.rename(final)
    report={"status":"PASS","dataset":"PerMIS","cases":216,"episodes":432,"images":648,
            "annotation_zip_sha256":args.expected_sha256,"test_annotation_sha256":digest(annotation),
            "source_manifest_sha256":digest(SOURCE/"SOURCE_MANIFEST.json"),"original_generator_sha256":digest(original),
            "local_generator_sha256":digest(patched),"raw_videos_after_official_filter":video_count,
            "frame_acquisition_manifest_sha256":digest(LOGS/"permis-frame-acquisition.json"),
            "frame_acquisition":acquisition_stats,"directory":str(final),
            "method":"Author masks and selection functions unchanged; output directory patched; image I/O shim retrieves original CRC-verified ZIP members. No resizing or custom track selection.",
            "metric":"Official DatasetPerMis: frame0 reference, frame1/frame2 targets, class_ids=[0]; use official evaluator."}
    (LOGS/"permis-data-check.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PERMIS_FULL_DATA_PASS",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
