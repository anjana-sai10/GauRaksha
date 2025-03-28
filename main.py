import torch
from torchvision import transforms
from PIL import Image

# Load Pre-trained AI Model (Replace with your trained model)
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

# Define Breed Labels
breed_labels = ["Gir", "Sahiwal", "Red Sindhi", "Kankrej", "Tharparkar", "Ongole", "Malnad Gidda"]

# Image Preprocessing
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict Breed
def predict_cow_breed(img_path):
    try:
        img_tensor = process_image(img_path)
        with torch.no_grad():
            output = model(img_tensor)
        breed_index = torch.argmax(output).item() % len(breed_labels)  # Dummy Mapping
        return f"🐄 The detected breed is: {breed_labels[breed_index]}"
    except Exception as e:
        return f"Error: {str(e)}"

# Example Usage
image_path = "test_cow.jpg"  # Replace with an actual cow image
print(predict_cow_breed(image_path))
