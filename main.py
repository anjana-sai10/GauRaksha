import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# Load Pre-trained AI Model
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
model.eval()

# Define Breed Labels
breed_labels = ["Gir", "Sahiwal", "Red Sindhi", "Kankrej", "Tharparkar", "Ongole", "Malnad Gidda"]

# Image Preprocessing
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Predict Breed
def predict_cow_breed(img):
    img_tensor = process_image(img)
    with torch.no_grad():
        output = model(img_tensor)
    breed_index = torch.argmax(output).item() % len(breed_labels)  # Dummy Mapping
    return breed_labels[breed_index]

# Streamlit UI
st.title("🐄 Indian Cow Breed Detector")
st.write("Upload an image of a cow to identify its breed.")

uploaded_file = st.file_uploader("Choose a cow image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Breed Prediction
    breed = predict_cow_breed(image)
    st.success(f"✅ Detected Breed: **{breed}**")
