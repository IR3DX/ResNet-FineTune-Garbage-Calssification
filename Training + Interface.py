# Train Model:------------------------------------------------------------------
# Install Libraries
!pip install -q kagglehub
!pip install -q torchvision torch torchvision torchaudio
!pip install -q gradio

# Import Libraries
import os
import torch
import torchvision
import kagglehub
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import gradio as gr
from PIL import Image

# Downlaod Data
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
print("Path to dataset files:", path)

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder("/kaggle/input/garbage-classification/Garbage classification/Garbage classification", transform=transform)

# Stratified split: 95% train, 5% test
targets = [sample[1] for sample in dataset.samples]
train_idx, test_idx = train_test_split(
    list(range(len(targets))),
    test_size=0.05,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet-18 and update final layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
device = torch.device("cuda")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}/{epochs}] Training Loss: {running_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), 'resnet18_trash_classifier.pth')
print("Model saved successfully!")

# Evaluate the model on the test set
model.eval() 
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        ThrowAway, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



# Gradio Interface:-------------------------------------------------------------

# Load the dataset to get class labels
class_names = dataset.classes

# Redefine the model structure
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet18_trash_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# preprocessing for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Main function
def classify_image(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, 0)
        predicted_label = class_names[predicted_idx]

    if confidence.item() < 0.3:
        return "None (Low Confidence)"

    return f"Predicted: {predicted_label} ({confidence.item()*100:.2f}%)"

# Get a few example image paths from the test dataset
example_images = []
for i in range(5):
    image_tensor, _ = test_dataset[i]
    image = transforms.ToPILImage()(image_path)
    image.save(f"example_{i}.png")
    example_images.append(f"example_{i}.png")

# Gradio Interface
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Trash Classifier with ResNet-18",
    description="Upload a trash image to classify it. If confidence is low, it returns 'None'.",
    examples=example_images
).launch()
