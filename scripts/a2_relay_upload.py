"""A2 (acc): relay the nvfp4-36l candidate (lora_kd_fp4a_rank64a128.pt baked in) and the shared NVFP4
experts side-files from Tokyo EFS to a scratch HF repo, so the Spain pod can pull them (no shared
storage between the two clusters). Same relay method as NVFP4-NEXT.md:14 (littlemex/prismyra-nvfp4-scratch),
own scratch repo to avoid clashing with any other concurrent use of that name."""
import sys
from huggingface_hub import HfApi

REPO = sys.argv[1] if len(sys.argv) > 1 else "littlemex/prismyra-nvfp4-scratch-acc"
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=True)
print("repo ready:", REPO)

api.upload_folder(repo_id=REPO, folder_path="/work/models/p35-l36-fp4a_rank64a128", path_in_repo="candidate", repo_type="model")
print("candidate uploaded")
api.upload_file(repo_id=REPO, path_or_fileobj="/work/next/models/nvfp4_experts_36l.safetensors", path_in_repo="nvfp4_experts_36l.safetensors", repo_type="model")
api.upload_file(repo_id=REPO, path_or_fileobj="/work/next/results/calib36.json", path_in_repo="calib36.json", repo_type="model")
print("side files uploaded")
