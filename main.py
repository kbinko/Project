"""
Główny plik projektu do trenowania modeli klasyfikacji obrazów.

Ten skrypt zawiera:
1. Definicje dwóch modeli: przetrenowanego modelu ResNet i prostego modelu CNN.
2. Przykładową pętlę treningową dla tych modeli.
3. Funkcje do wczytywania i przetwarzania danych z wykorzystaniem klasy CustomDataset.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from torchvision import transforms

# 2. Nieprzetrenowany Model CNN
class SimpleCNN(nn.Module):
    """
    Prosta konwolucyjna sieć neuronowa (CNN).

    Zawiera dwie warstwy konwolucyjne i dwie warstwy w pełni połączone.
    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 1. Przetrenowany Model ResNet
def get_pretrained_resnet(num_classes):
    """
    Funkcja do tworzenia przetrenowanego modelu ResNet z dostosowaną ostatnią warstwą.
    Zwraca model ResNet dostosowany do zadanej liczby klas.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Przykładowe użycie modeli
num_classes = 2  # Liczba klas (samochód, brak samochodu)
pretrained_resnet = get_pretrained_resnet(num_classes)
simple_cnn = SimpleCNN(num_classes)

# Definiowanie transformacji dla danych
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Przygotowanie danych
dataset = CustomDataset(images_dir='data/obrazy', labels_dir='data/etykiety', transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Definiowanie funkcji straty i optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_resnet.parameters(), lr=0.001)

# Pętla treningowa 
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        # Zerowanie gradientów
        optimizer.zero_grad()

        # Przekazywanie danych do modelu
        outputs = pretrained_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Obliczanie straty
        running_loss += loss.item()
    print(f"Epoka {epoch + 1} - średnia strata: {running_loss / len(data_loader)}")
print('Trenowanie zakończone')