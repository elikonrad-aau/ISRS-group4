import json
import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from torchvision import models
from transformers import AutoImageProcessor,AutoModel,CLIPModel,CLIPProcessor


#
# helper functions
#
# saving the embeddings
def save_embeddings(model_name, movie_ids, movie_embeddings, output_dir):
    # create output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # convert into numpy array and save
    movie_embeddings = np.array(movie_embeddings, dtype=np.float32)
    np.save(model_output_dir / "movie_embeddings.npy", movie_embeddings)

    # save movieId to the corresponding embeddings
    with open(model_output_dir / "movie_ids.json", "w", encoding="utf-8") as file:
        json.dump(movie_ids, file, indent=2)


# create embeddings from images
def create_embeddings(model_name, model, processor, movie_images, output_dir, device, batch_size=32, uses_huggingface=False):
    print(f"Creating embeddings with {model_name}:")

    # copy model to device (faster processing outside docker)
    model = model.to(device)
    model.eval()

    movie_ids = []
    movie_embeddings = []

    # creating a progress indicator
    total_images = sum(len(paths) for paths in movie_images.values())
    progress = tqdm(total=total_images, desc=f"{model_name} images", leave=True)

    # creating the embeddings (disable gradient tracking)
    with torch.no_grad():
        for movie_id, image_paths in tqdm(movie_images.items(), desc=f"Embeddings with {model_name}"):
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
                    finally:
                        progress.update(1)

                if not batch_images:
                    continue

                # add-on for huggingface models (clip, dinov3)
                if uses_huggingface:
                    inputs = processor(images=batch_images, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}

                    outputs = model(**inputs)

                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        batch_embeddings = outputs.pooler_output
                    else:
                        batch_embeddings = outputs.last_hidden_state[:, 0, :]
                # normal version for resnet
                else:
                    batch_tensors = [processor(image) for image in batch_images]
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    batch_embeddings = model(batch_tensor)

                image_embeddings.extend(batch_embeddings.cpu().numpy())

            if not image_embeddings:
                continue

            # create movie embedding (multiple backdrops per movie)
            movie_embedding = np.mean(image_embeddings, axis=0)

            # normalization - cosine similarity
            norm = np.linalg.norm(movie_embedding)
            if norm > 0:
                movie_embedding = movie_embedding / norm

            # create lists
            movie_ids.append(movie_id)
            movie_embeddings.append(movie_embedding)

    # progress (tqdm)
    progress.close()

    # save embedding to files
    save_embeddings(model_name=model_name, movie_ids=movie_ids, movie_embeddings=movie_embeddings, output_dir=output_dir)


#
# resnet50 embeddings
#
def create_resnet50_embeddings(movie_images, output_dir, device):
    # load model
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # remove classifier – output becomes image embedding
    model.fc = torch.nn.Identity()

    # create embedding
    create_embeddings("resnet50", model, weights.transforms(), movie_images, output_dir, device)


#
# clip embeddings
#
def create_clip_embeddings(movie_images, output_dir, device):
    # get huggingface token from .env file
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Token is missing from the .env file.")

    # model name (huggingface)
    clip_models = {
        "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
        "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    }

    # iterate over huggingface models
    for output_name, hf_model_name in clip_models.items():
        # load model
        processor = AutoImageProcessor.from_pretrained(hf_model_name, token=hf_token)
        model = AutoModel.from_pretrained(hf_model_name, token=hf_token)

        # only use the vision model
        model = model.vision_model

        # create image embeddings
        create_embeddings(output_name, model, processor, movie_images, output_dir, device, uses_huggingface=True)

        # cleanup for next model
        del model
        torch.cuda.empty_cache()


#
# dinov3 embeddings
#
def create_dinov3_embeddings(movie_images, output_dir, device):
    # get huggingface token from .env file
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Token is missing from the .env file.")

    # model name (huggingface)
    dinov3_models = {
        "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }

    # iterate over huggingface models
    for output_name, hf_model_name in dinov3_models.items():
        # load model
        processor = AutoImageProcessor.from_pretrained(hf_model_name, token=hf_token)
        model = AutoModel.from_pretrained(hf_model_name, token=hf_token)

        # create image embeddings
        create_embeddings(output_name, model, processor, movie_images, output_dir, device, uses_huggingface=True)

        # cleanup for next model
        del model
        torch.cuda.empty_cache()


#
# main procedure
#
def main():
    # get project root / works outside django
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

    # bookkeeping
    print("Creating embeddings...")
    print("Device:", device)
    print("Movies found:", len(movie_images))
    print("Images found:", sum(len(paths) for paths in movie_images.values()))

    # create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # run the following embeddings
    create_resnet50_embeddings(movie_images, output_dir, device)
    create_clip_embeddings(movie_images, output_dir, device)
    create_dinov3_embeddings(movie_images, output_dir, device)


if __name__ == "__main__":
    main()