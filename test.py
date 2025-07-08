import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from torchvision import transforms

# Load feature embeddings and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load FaceNet model (512-d embeddings)
model = InceptionResnetV1(pretrained='vggface2').eval()

# MTCNN face detector
detector = MTCNN()

# Load sample image and detect face
sample_img = cv2.imread('sample/Ranbir_Kapoor.61.jpg')
sample_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(sample_rgb)

if results:
    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)  # Ensure no negative indices
    face = sample_rgb[y:y + height, x:x + width]

    # Preprocess face
    image = Image.fromarray(face).resize((160, 160))  # FaceNet expects 160x160
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 160, 160)

    with torch.no_grad():
        result = model(img_tensor).squeeze().numpy()  # Shape: (512,)

    # Cosine similarity
    similarity = [cosine_similarity(result.reshape(1, -1), feat.reshape(1, -1))[0][0] for feat in feature_list]

    # Get the most similar image
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    # Show the matched image
    temp_img = cv2.imread(filenames[index_pos])
    cv2.imshow('Matched Celebrity', temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No face detected in the image.")
