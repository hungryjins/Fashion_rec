from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from image_utils import extract_img_features
from transformers import CLIPProcessor, CLIPModel


if __name__ == "__main__":
    # load CLIP
    model_name = "patrickjohncyh/fashion-clip"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("-"*60)
    print("CLIP model loading completed, starting embedding extraction")

    # Load paths of cropped images
    cropped_path = "imaterialist-fashion-2020-fgvc7/cropped_images"
    images = list(os.walk(cropped_path))[0][2]
    images = [i for i in images if '.jpg' in i]

    # Delete if existing data exists
    open('img_embeddings_fashion_fine_tuned.json', 'w').close()

    # Save locally line by line
    for i in tqdm(images):
        img = Image.open(os.path.join(cropped_path, i))
        # Extract embeddings using CLIP
        img_emb = extract_img_features(img, processor, model)
        # Write to json file line by line (to prevent loading embeddings in RAM)
        with open('img_embeddings_fashion_fine_tuned.json','a') as file:
            key = i.split(".")[0]
            d = {key : np.array(img_emb)[0].tolist()}
            json_string = json.dumps(d, ensure_ascii=False)
            file.write(json_string + '\n')
