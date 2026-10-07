"""acc3: relay the a8-nvfp4-36l candidate (routed experts stripped, byte-level, same layout as the public
nvfp4-36l repo) from Tokyo EFS to a scratch private HF repo, so the Spain pod can pull it down.
"""
import sys
from huggingface_hub import HfApi

REPO = sys.argv[1] if len(sys.argv) > 1 else "littlemex/prismyra-nvfp4-scratch-acc3"
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=True)
print("repo ready:", REPO)
api.upload_folder(repo_id=REPO, folder_path="/work/next/models/p35-l36-kd_a8-nvfp4-noexp", path_in_repo="a8-noexp", repo_type="model")
print("a8-noexp uploaded")
