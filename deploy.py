"""Deploy digital-clone-space/ to a public Hugging Face Space.

    .venv/Scripts/python deploy.py              # -> usman-07/usman-digital-clone
    .venv/Scripts/python deploy.py my-org/name  # e.g. a class organisation

Reads Huggingface_Access_token and GEMINI_API_KEY from .env, stores them as
Space *secrets* (never uploaded as files), then uploads the app folder.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

token = env["Huggingface_Access_token"]
api = HfApi(token=token)
repo = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/usman-digital-clone"

print("creating", repo)
api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                private=False, exist_ok=True)
api.add_space_secret(repo, "GEMINI_API_KEY", env["GEMINI_API_KEY"])
api.add_space_secret(repo, "HF_TOKEN", token)
api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT / "digital-clone-space"),
                  ignore_patterns=["__pycache__/*", "*.pyc", ".env", "*.env"],
                  commit_message="Digital clone: hybrid RAG app, knowledge base, report")
print(f"done -> https://huggingface.co/spaces/{repo}")
