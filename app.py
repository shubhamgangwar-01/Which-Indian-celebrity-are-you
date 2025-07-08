import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms

# Initialize model and face detector
detector = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load precomputed embeddings and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Ensure upload directory exists
os.makedirs('uploads', exist_ok=True)

# Save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

# Extract features from face image using MTCNN + FaceNet
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not results:
        return None

    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img_rgb[y:y + height, x:x + width]

    image = Image.fromarray(face).resize((160, 160))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    img_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 160, 160)

    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()

    return embedding

# Recommend most similar celebrity
def recommend(feature_list, features):
    similarity = [
        cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0]
        for f in feature_list
    ]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit UI
st.title('✨ Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('📸 Upload a clear face image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        img_path = os.path.join('uploads', uploaded_image.name)

        # Extract features
        features = extract_features(img_path, model, detector)

        if features is not None:
            index_pos = recommend(feature_list, features)
            # Extract just the filename (no path)
            filename = os.path.basename(filenames[index_pos])

            # Remove extension and split by "_"
            name_without_ext = os.path.splitext(filename)[0]  
            predicted_actor = " ".join(name_without_ext.split('_')) 

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Your Image')
                st.image(display_image, use_container_width=True)

            with col2:
                st.subheader(f"You look like {predicted_actor}")
                st.image(filenames[index_pos], use_container_width=True)

        else:
            st.warning("No face detected. Please upload a clearer image.")
