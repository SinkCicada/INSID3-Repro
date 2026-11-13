"""Fetch and verify the checkpoints used by this reproduction."""
import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

BASE = "https://github.com/SinkCicada/INSID3-Repro/releases/download/weights-v1/"
MODELS = {
    "large": ("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"),
    "base": ("dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"),
    "small": ("dinov3_vits16_pretrain_lvd1689m-08c60483.pth", "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"),
    "sam3": ("sam3.pt", "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"),
    "dinov2-base": ("dinov2_vitb14_pretrain.pth", "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"),
    "dinov2-large": ("dinov2_vitl14_pretrain.pth", "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"),
    "franca": ("franca_vitb14_In21k.pth", "2a0888b5bd5be646704f1ecb8409913a0321db97d0cad00df1e924fd38cf6d2a"),
    "pe": ("PE-Spatial-L14-448.pt", "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1"),
    "sam": ("sam_vit_h_4b8939.pth", "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"),
    "seggpt": ("seggpt_vit_large.pth", "59c74d1993ca50ac0ae968d9bbe5db453f50aa5d5c7c9564cce6011c18b16570"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(stream)


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", choices=MODELS, default=["large"])
    group.add_argument("--all", action="store_true", help="Fetch every reproduction checkpoint")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "pretrain")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for model in MODELS if args.all else args.models:
        name, expected = MODELS[model]
        target = args.directory / name
        if target.exists():
            if digest(target) != expected:
                parser.error(f"Existing file has a different SHA-256: {target}")
            print(f"Verified existing {target}", flush=True)
            continue
        parts = [name + ".part01", name + ".part02"] if model in {"sam", "sam3"} else [name]
        temp = target.with_name(target.name + ".download")
        try:
            with temp.open("wb") as stream:
                for part in parts:
                    print(f"Downloading {part}", flush=True)
                    with urlopen(BASE + part, timeout=60) as response:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
            if digest(temp) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    main()
