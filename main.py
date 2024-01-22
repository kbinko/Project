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
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from custom_dataset import CustomDataset
from torchvision.models import resnet50, ResNet50_Weights, alexnet, AlexNet_Weights, vgg16, VGG16_Weights, squeezenet1_0, SqueezeNet1_0_Weights, densenet161, DenseNet161_Weights
from sklearn.metrics import precision_score, recall_score, f1_score

#Załadowanie przetrenowanych modeli

#Model resnet50
model_resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs, 1)

# Model AlexNet
model_alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
num_ftrs = model_alexnet.classifier[6].in_features
model_alexnet.classifier[6] = nn.Linear(num_ftrs, 1)

# Model VGG16
model_vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
num_ftrs = model_vgg16.classifier[6].in_features
model_vgg16.classifier[6] = nn.Linear(num_ftrs, 1)

# Model SqueezeNet
model_squeezenet = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
model_squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1))
model_squeezenet.num_classes = 1

# Model DenseNet
model_densenet = densenet161(weights=DenseNet161_Weights.DEFAULT)
num_ftrs = model_densenet.classifier.in_features
model_densenet.classifier = nn.Linear(num_ftrs, 1)

# Załadowanie danych 
transform = transforms.Compose([    
    transforms.Resize(256),   #obraz 256x256     
    transforms.ToTensor(), #konwersja obrazu do formatu tensora             
    transforms.Normalize(              
    mean=[0.485, 0.456, 0.406],        
    std=[0.229, 0.224, 0.225] # normalizacja obrazu na podstawie średniej i odchylenia standardowego          
 )])
dataset = CustomDataset('data/obrazy', 'data/etykiety', transform=transform)

# Podział danych na treningowe i walidacyjne
train_size = int(0.65 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted.flatten() == labels.flatten()).sum().item()

            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = 100 * correct / total
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return accuracy, precision, recall, f1

# Trening własnego modelu
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 128 * 128, 128) 
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Spłaszczanie tensora
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x
    
# Inicjalizacja modelu, funkcji straty i optymalizatora
model = SimpleCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pętla treningowa
"""
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Walidacja
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoka {epoch+1}, Strata treningowa: {(running_loss / len(train_loader)):.4f}, Strata walidacyjna: {(val_loss / len(val_loader)):.4f}")

print("Trening zakończony")
"""

models_list = [model_resnet50, model_alexnet, model_vgg16, model_squeezenet, model_densenet, model]
for model in models_list:
    accuracy, precision, recall, f1 = test_model(model, test_loader)
    print(f"Model: {model.__class__.__name__}")
    print(f"Dokładność: {(accuracy):.2f}%")
    print(f"Precyzja: {(precision):.2f}")
    print(f"Czułość: {(recall):.2f}")
    print(f"Miara F1: {(f1):.2f}")