import os
import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_DIR = "C:/Users/elisa/PycharmProjects/ISRS-group4/dataset/subtitles"
OUTPUT_DIR = "./saved_model"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 200  # Split text files (subtitles) in 200 words to avoid token limit 256


def clean_srt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
    content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'<[^>]+>', '', content)

    lines = [line.strip() for line in content.split('\n') if line.strip()]
    return " ".join(lines)


def extract_movie_id(filename):
    match = re.match(r'^(\d+)\.srt$', filename)
    return int(match.group(1)) if match else None


def split_into_chunks(text, chunk_size_words):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = " ".join(words[i: i + chunk_size_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def main():
    model = SentenceTransformer(MODEL_NAME)
    input_path = Path(INPUT_DIR)
    srt_files = sorted(input_path.glob('*.srt'))
    movie_ids = []
    final_embeddings = []

    for filepath in tqdm(srt_files, desc="Processing Movies"):
        mid = extract_movie_id(filepath.name)
        if mid is None:
            continue

        raw_text = clean_srt(filepath)
        if len(raw_text.split()) < 10:
            continue

        chunks = split_into_chunks(raw_text, CHUNK_SIZE_WORDS)

        if not chunks:
            continue

        chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        # Average to get ONE vector for the whole movie subtitles
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        movie_ids.append(mid)
        final_embeddings.append(avg_embedding)

    embeddings_matrix = np.array(final_embeddings)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = output_path / "model"
    model.save(str(model_path))
    np.save(output_path / "vectors.npy", embeddings_matrix)

    with open(output_path / "movie_ids.json", 'w') as f:
        json.dump(movie_ids, f)

    id_to_idx = {str(m): i for i, m in enumerate(movie_ids)}
    with open(output_path / "id_to_index.json", 'w') as f:
        json.dump(id_to_idx, f)

    meta = {
        "model": MODEL_NAME,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "dim": embeddings_matrix.shape[1],
        "count": len(movie_ids),
        "created": timestamp
    }
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()