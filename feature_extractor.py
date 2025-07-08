import os
import pickle

actors = os.listdir('data')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))

pickle.dump(filenames, open('filenames.pkl','wb'))

import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import numpy as np
import pickle
from tqdm import tqdm

## Load image file paths
filenames = pickle.load(open('filenames.pkl', 'rb'))

## Pretrained Facenet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Facenet expects 160x160
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Feature extractor function
def feature_extractor(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 160, 160)

    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()  # Shape: (512,)

    return embedding

# Extract features
features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

# Save the features
pickle.dump(features, open('embedding.pkl', 'wb'))



