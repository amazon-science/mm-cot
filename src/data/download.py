import os
import zipfile

import requests
from tqdm import tqdm


def unzip(file, dest):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def download_file(url, file_name):
    response = requests.get(url, stream=True)
    print(response)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(file_name, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            tqdm_bar.update(len(data))
    tqdm_bar.close()


# # Download data
os.system("git clone https://github.com/lupantech/ScienceQA")
os.system("mv ScienceQA/data data")
os.system("rm -r ScienceQA")

# Download vision features
download_file(
    "https://drive.google.com/u/0/uc?id=13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr&export=download&confirm=t&uuid=dde4f5c5-a182-41fb-b908-934bb153d5e7&at=ALgDtsyO45szTP40cSRZbVR9h8iU:1677195067586",
    "vision_features.zip")

os.mkdir('vision_features')
unzip('vision_features.zip', '.')

# Download models
download_file(
    'https://drive.google.com/u/0/uc?id=1FtTYOJPHnWnFfCxNC6M3gar4RAX5E21b&export=download&confirm=t&uuid=a5d17c71-252a-421b-9fd0-6ce609da7947&at=ALgDtsxrRDAAnvFetgzK3lXuZ9I_:1676314225641',
    'models.zip'
)

os.mkdir('models')
unzip('models.zip', '.')



# FAKEDDIT

url_test = "https://drive.google.com/u/0/uc?id=1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv&export=download"
url_train = "https://drive.google.com/u/0/uc?id=1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6&export=download&confirm=t&uuid=0ecba788-1cbc-4ec1-85a9-df179550b905&at=ALgDtsxuebBBCouh_mamrytmUe3_:1677695702113"
url_validate = "https://drive.google.com/u/0/uc?id=1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv&export=download"

download_file(url_test, "data/fakeddit/full/multimodal_test_public.tsv")
download_file(url_train, "data/fakeddit/full/multimodal_train_public.tsv")
download_file(url_validate, "data/fakeddit/full/multimodal_validate_public.tsv")

import pandas as pd

SAMPLE_NUMBER = 10000
RANDOM_STATE = 1

df = pd.read_csv("data/fakeddit/full/multimodal_train_public.tsv", delimiter="\t")
df_sub_sample = df.sample(n=SAMPLE_NUMBER, random_state=RANDOM_STATE)

df_sub_sample.to_csv("data/fakeddit/partial/dataset.csv",)

from multiprocessing import Pool
import requests

def process(row):
  id = row["id"]
  path = f"data/fakeddit/images/{id}.jpg"
  url = row["image_url"]
  
  if type(url) != float and f"{id}.jpg":
    response =  requests.get(url)
    if response.status_code == 200:
      img_data = response.content
      with open(path, 'wb') as handler:
          handler.write(img_data)

n = 200  # Any number of threads
with Pool(n) as p:
    p.map(process, df_sub_sample.to_dict(orient="records"))

import shutil
shutil.make_archive('data/fakeddit/images', 'zip', 'data/fakeddit/images')
rm -r data/fakeddit/images