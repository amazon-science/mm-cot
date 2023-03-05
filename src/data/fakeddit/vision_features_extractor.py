import os
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image
from src import constants
import pandas as pd

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101-dc5")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101-dc5")
image_features = []

dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)

for index, row in enumerate(dataframe.to_dict(orient="records")[:10]):
    image_feature = {}
    image_feature["index"] = index

    if row["image_url"]:
        try:
            image_path = os.path.join(constants.FAKEDDIT_IMG_DATASET_PATH, f"{row['id']}.jpg")
            image = Image.open(image_path)
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs) 

            # the last hidden states are the final query embeddings of the Transformer decoder
            # these are of shape (batch_size, num_queries, hidden_size)
            last_hidden_states = outputs.last_hidden_state
            
            image_feature["features"] = last_hidden_states 
        except FileNotFoundError:
            pass

    else:
        image_feature["features"] = []

    image_features.append(image_feature)
