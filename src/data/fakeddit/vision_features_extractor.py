import os
import sys
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image, UnidentifiedImageError
from src import constants
import pandas as pd
import torch
import numpy as np

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101-dc5")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101-dc5")
image_features = []

try:
    image_features = np.load(constants.FAKEDDIT_VISION_FEATURES_PATH, allow_pickle=True).tolist()
except FileNotFoundError:
    pass

checkpoint = len(image_features)
dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)

with torch.no_grad():
    for index, row in enumerate(dataframe.to_dict(orient="records")[checkpoint:]):
        image_feature = np.array([])

        print(f"PROCESSING: {row['id']}")
        if row['image_url']:
            try:
                image_path = os.path.join(constants.FAKEDDIT_IMG_DATASET_PATH, f"{row['id']}.jpg")
                image = Image.open(image_path)
                inputs = image_processor(images=image, return_tensors="pt")
                outputs = model(**inputs) 

                # the last hidden states are the final query embeddings of the Transformer decoder
                # these are of shape (batch_size, num_queries, hidden_size)
                image_feature = outputs.last_hidden_state.numpy()

            except (FileNotFoundError,  ValueError, UnidentifiedImageError) as err:
                print(f"{row['id']} || {err}")

        image_features.append(image_feature)
        np.save("data/fakeddit/partial/vision_features/image_features", np.asarray(image_features))
