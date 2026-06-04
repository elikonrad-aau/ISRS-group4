import os
import numpy as np
import torch
import json

from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from image_similarity import save_embeddings


#
# helper functions
#
# load movie genomes from json (works outside of docker for gpu/mpu)
def load_movie_genome_tags():
    # find file
    profiles_path = Path(__file__).resolve().parent / "movie_genome_profiles.json"

    # load file
    with open(profiles_path, "r", encoding="utf-8") as file:
        movie_tags = json.load(file)

    return {
        str(movie_id): [(str(tag), float(relevance)) for tag, relevance in tag_items] for movie_id, tag_items in movie_tags.items()
    }


# create embeddings from images and genome tags
def create_embeddings(model_name, model, processor, movie_images, movie_tags, output_dir, device, batch_size=16, image_weight=0.7, text_weight=0.3):
    print(f"Creating embeddings with {model_name}:")

    # copy model to device
    model = model.to(device)
    model.eval()

    movie_ids = []
    movie_embeddings = []

    # create union of movies with images and genome tags
    all_movie_ids = sorted(set(movie_images.keys()) | set(movie_tags.keys()))

    # creating a progress indicator
    progress = tqdm(total=len(all_movie_ids), desc=f"{model_name} movies", leave=True)

    # creating the embeddings (disable gradient tracking)
    with torch.no_grad():
        for movie_id in all_movie_ids:
            embedding_parts = []

            #
            # image embeddings
            image_paths = movie_images.get(movie_id, [])
            image_embeddings = []

            # iterate over images in batches
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                batch_images = []

                # iterating over images in the batch
                for image_path in batch_paths:
                    try:
                        batch_images.append(Image.open(image_path).convert("RGB"))
                    except Exception as e:
                        print(e)

                if not batch_images:
                    continue

                # process image batch
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # create image embeddings
                batch_embeddings = model.vision_model(
                    pixel_values=inputs["pixel_values"],
                    return_dict=True,
                ).pooler_output

                batch_embeddings = model.visual_projection(batch_embeddings)

                # normalization - cosine similarity
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

                image_embeddings.extend(batch_embeddings.cpu().numpy())

            if image_embeddings:
                # create image embedding (multiple backdrops per movie)
                image_embedding = np.mean(image_embeddings, axis=0)

                # normalization - cosine similarity
                norm = np.linalg.norm(image_embedding)
                if norm > 0:
                    image_embedding = image_embedding / norm

                embedding_parts.append((image_weight, image_embedding))

            #
            # genome tag embeddings (text embeddings)
            tag_items = movie_tags.get(movie_id, [])

            if tag_items:
                # convert genome tags into text
                tag_texts = [str(tag) for tag, relevance in tag_items]

                # use relevance as weight
                tag_weights = np.array([relevance for tag, relevance in tag_items], dtype=np.float32)

                # process text batch
                inputs = processor(text=tag_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # create text embeddings
                text_embeddings = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                ).pooler_output

                text_embeddings = model.text_projection(text_embeddings)

                # normalization - cosine similarity
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                text_embeddings = text_embeddings.cpu().numpy()

                # create genome embedding (multiple tags per movie)
                tag_weights = tag_weights / tag_weights.sum()
                text_embedding = np.average(text_embeddings, axis=0, weights=tag_weights)

                # normalization - cosine similarity
                norm = np.linalg.norm(text_embedding)
                if norm > 0:
                    text_embedding = text_embedding / norm

                embedding_parts.append((text_weight, text_embedding))

            if not embedding_parts:
                progress.update(1)
                continue

            # combine image and genome embeddings
            total_weight = sum(weight for weight, embedding in embedding_parts)
            movie_embedding = sum((weight / total_weight) * embedding for weight, embedding in embedding_parts)

            # normalization - cosine similarity
            norm = np.linalg.norm(movie_embedding)
            if norm > 0:
                movie_embedding = movie_embedding / norm

            # create lists
            movie_ids.append(movie_id)
            movie_embeddings.append(movie_embedding)

            progress.update(1)

    # progress (tqdm)
    progress.close()

    # save embedding to files
    save_embeddings(model_name=model_name, movie_ids=movie_ids, movie_embeddings=movie_embeddings, output_dir=output_dir)


#
# clip embeddings
#
def create_clip_embeddings(movie_images, movie_tags, output_dir, device):
    # get huggingface token from .env file
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Token is missing from the .env file.")

    # model name (huggingface)
    clip_models = {
        "clip-vit-large-patch14-image-genome": "openai/clip-vit-large-patch14",
    }

    # iterate over huggingface models
    for output_name, hf_model_name in clip_models.items():
        # load model
        processor = CLIPProcessor.from_pretrained(hf_model_name, token=hf_token)
        model = CLIPModel.from_pretrained(hf_model_name, token=hf_token)

        # create image and genome embeddings
        create_embeddings(output_name, model, processor, movie_images, movie_tags, output_dir, device)

        # cleanup for next model
        del model
        torch.cuda.empty_cache()


#
# main procedure
#
def main():
    # get project root / works inside django
    project_root = Path(__file__).resolve().parents[3]

    # parse HF_TOKEN from environment
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

    # store downloaded models from hugging face in project
    os.environ["HF_HOME"] = str(project_root / "dataset" / "models" / "huggingface")

    # define directories
    backdrops_dir = project_root / "dataset" / "tmdb" / "resnet_backdrops"
    output_dir = project_root / "apps" / "recommender" / "embeddings"

    # check the running environment for faster processing
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # create dictionary
    movie_images = defaultdict(list)

    # possible image extensions for the backdrops
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # build the movie directory once and reuse
    for movie_dir in backdrops_dir.iterdir():
        if not movie_dir.is_dir():
            continue

        for image_path in movie_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                movie_images[movie_dir.name].append(image_path)

    movie_images = {
        movie_id: sorted(paths)
        for movie_id, paths in sorted(movie_images.items())
    }

    # load genome tags
    movie_tags = load_movie_genome_tags()

    # bookkeeping
    print("Creating embeddings...")
    print("Device:", device)
    print("Movies found:", len(movie_images))
    print("Images found:", sum(len(paths) for paths in movie_images.values()))
    print("Genome profiles found:", len(movie_tags))

    # create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # run the following embeddings
    create_clip_embeddings(movie_images, movie_tags, output_dir, device)


if __name__ == "__main__":
    main()