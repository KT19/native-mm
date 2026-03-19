import os

import webdataset as wds
from datasets import load_dataset
from tqdm import tqdm


def convert_to_wds(dataset_name, output_dir, limit=500000, shard_size=10000):
    os.makedirs(output_dir, exist_ok=True)

    # Pattern for tar file name
    sink = wds.ShardWriter(f"{output_dir}/shard-%05d.tar", maxcount=shard_size)  # type: ignore

    print(f"Converting {limit} samples to {output_dir}...")
    try:
        # Load from HF Hub (streaming=True to avoid full download before slicing)
        ds = load_dataset(dataset_name, split="train", streaming=True)
        for i, ex in enumerate(tqdm(ds)):
            if i >= limit:
                break

            image = ex.get("image")
            caption = ex.get("caption")

            if image is None or caption is None:
                continue

            # Write to shard
            sink.write({"__key__": f"sample_{i:08d}", "jpg": image, "txt": caption})
    except Exception as e:
        print(f"An error occured: {e}")
    finally:
        sink.close()
        print("Done!")


if __name__ == "__main__":
    convert_to_wds(
        dataset_name="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M", output_dir="./local_llava_data", limit=300000
    )
