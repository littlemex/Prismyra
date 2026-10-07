import os
os.environ.setdefault("HF_HOME", "/models/.hf")
from huggingface_hub import snapshot_download
print("downloading public nvfp4-36l (L1 baseline + nvfp4 side files)...")
p1 = snapshot_download("littlemex/prismyra-decision-qwen3.6-35b-a3b-nvfp4-36l", local_dir="/models/nvfp4-36l-public")
print("public ->", p1)
print("downloading a8 scratch (non-expert, byte-stripped shards)...")
p2 = snapshot_download("littlemex/prismyra-nvfp4-scratch-acc3", local_dir="/models/a8-scratch")
print("scratch ->", p2)
