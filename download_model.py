from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B",
    local_dir="./models/Qwen3-Embedding-0.6B",
)

print("Download complete.")