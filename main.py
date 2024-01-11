import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform
        self.images = [file for file in os.listdir(images_folder) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        label_path = os.path.join(self.labels_folder, os.path.splitext(img_name)[0] + '.json')
        with open(label_path, 'r') as f:
            label = json.load(f)

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformacje
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Utworzenie instancji Dataset i DataLoader
dataset = CustomDataset('data/obrazy', 'data/etykiety', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Przykładowe użycie DataLoadera
for images, labels in data_loader:
    # Tutaj można umieścić logikę treningu modelu
    pass

# Tutaj możesz dodać kod do definicji i treningu modelu
