import os

import webdataset as wds
from datasets import load_dataset
from tqdm import tqdm


def convert_to_wds(dataset_name, output_dir, limit=500000, shard_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    # Load from HF Hub (streaming=True to avoid full download before slicing)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Pattern for tar file name
    sink = wds.ShardWriter(f"{output_dir}/shard-%05d.tar", maxcount=shard_size)  # type: ignore

    print(f"Converting {limit} samples to {output_dir}...")

    for i, ex in enumerate(tqdm(ds)):
        if i >= limit:
            break

        image = ex.get("image")
        caption = ex.get("caption")

        if image is None or caption is None:
            continue

        # Write to shard
        sink.write({"__key__": f"sample_{i:08d}", "jpg": image, "txt": caption})

    sink.close()
    print("Done!")
    del ds
    import gc

    gc.collect()


if __name__ == "__main__":
    convert_to_wds(
        dataset_name="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M", output_dir="./local_llava_data", limit=100000
    )
