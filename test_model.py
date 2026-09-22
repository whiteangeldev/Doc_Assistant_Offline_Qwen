import os

# Prevent Hugging Face network access.
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "./models/Qwen3-Embedding-0.6B",
    local_files_only=True,
    device="cpu",
)

documents = [
    "A blocked cooling fan can cause the motor to overheat.",
    "The contract expires in December.",
]

document_vectors = model.encode(
    documents,
    normalize_embeddings=True,
)

query_vector = model.encode(
    ["What causes the motor to get too hot?"],
    prompt_name="query",
    normalize_embeddings=True,
)

scores = model.similarity(query_vector, document_vectors)[0]

for document, score in zip(documents, scores):
    print(f"{float(score):.3f} | {document}")
