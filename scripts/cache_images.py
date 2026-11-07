"""Make resized copies (short side = 256, JPEG q95) in data/cache256 with the same folder layout."""
import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_ROOT, PROJECT_ROOT, SPLIT_DIRS, format_time, list_images


def resize_one(rel_path, src_root, dst_root, size, quality, overwrite):
    dst = os.path.join(dst_root, os.path.splitext(rel_path)[0] + ".jpg")
    if os.path.exists(dst) and not overwrite:
        return "skip"
    try:
        with Image.open(os.path.join(src_root, rel_path)) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.BICUBIC)
            im.save(dst, "JPEG", quality=quality)
        return "ok"
    except Exception as e:
        print(f"failed {rel_path}: {e!r}")
        return "fail"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--out", default=None, help="default: data/cache<size>")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    out = args.out or os.path.join(PROJECT_ROOT, "data", f"cache{args.size}")
    files = []
    for d in SPLIT_DIRS:
        os.makedirs(os.path.join(out, d), exist_ok=True)
        files += [f"{d}/{n}" for n in list_images(os.path.join(DATA_ROOT, d))]

    fn = partial(resize_one, src_root=DATA_ROOT, dst_root=out, size=args.size,
                 quality=args.quality, overwrite=args.overwrite)
    with Pool(args.workers) as pool:
        status = list(tqdm(pool.imap(fn, files, chunksize=64), total=len(files)))

    for s in ["ok", "skip", "fail"]:
        print(f"{s}: {status.count(s)}")
    for d in SPLIT_DIRS:
        print(f"  {d:10s} {len(list_images(os.path.join(out, d)))}")
    print(f"Output: {out}")
    print(f"Total time: {format_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
