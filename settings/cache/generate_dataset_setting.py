


import math
import os

# -----------------------
# 参数（可调）
# -----------------------
START_IDX = 0        # cc12m-train-0001
END_IDX = 650       # cc12m-train-2174（包含）
K = 30               # 生成多少个 toml

OUTPUT_DIR = "/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/dataset"
BASE_PATH = "/inspire/hdd/project/chineseculture/public/yuxuan/datasets/cc12m-unpacked"
PREFIX = "cc12m-train-"

# -----------------------
# 派生参数
# -----------------------
N = END_IDX - START_IDX + 1
chunk_size = math.ceil(N / K)

print(f"Dirs: [{START_IDX:04d} - {END_IDX:04d}]")
print(f"Total dirs: {N}, Shards: {K}, Chunk size: {chunk_size}")

# -----------------------
# 创建输出目录
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 生成 toml
# -----------------------
for shard_id in range(K):
    local_start = shard_id * chunk_size
    local_end = min(N, (shard_id + 1) * chunk_size)

    if local_start >= local_end:
        break

    global_start = START_IDX + local_start
    global_end = START_IDX + local_end - 1

    toml_path = os.path.join(
        OUTPUT_DIR,
        f"sd3_cache_dataset_{shard_id:04d}.toml"
    )

    with open(toml_path, "w") as f:
        f.write("resolutions = [256]\n\n")

        for idx in range(global_start, global_end + 1):
            dirname = f"{PREFIX}{idx:04d}"
            full_path = os.path.join(BASE_PATH, dirname)

            f.write("[[directory]]\n")
            f.write(f"path = '{full_path}'\n")
            f.write("num_repeats = 1\n\n")

    print(
        f"Generated: {toml_path} "
        f"({global_start:04d} - {global_end:04d})"
    )

print("DONE.")
