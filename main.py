import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import torch.nn as nn
import torch.optim as optim


# Utworzenie własnej klasy Dataset odpowiadającej temu, jak wygląda zbiór danych
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
        # sprawdzenie czy istnieje plik z etykietą
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
        else: 
            label = [] # pusta lista współrzędnych jeśli plik z etykietą nie istnieje

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

# Inicjalizacja i dostowanie modelu ResNet18
model = models.resnet18(pretrained=True) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # 2 klasy (samochod, brak samochodu)

# Definiowanie funkcji straty i optymalizatora w celu trenowania modelu
criterion = nn.CrossEntropyLoss() # Funkcja straty
optimizer = optim.Adam(model.parameters(), lr=0.001) # lr - learning rate

# Trenowanie modelu
def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train() # Ustawienie modelu w tryb trenowania
    for epoch in range(num_epochs): 
        for images, labels in data_loader:
            
            # Przygotowanie danych
            optimizer.zero_grad() # Wyzerowanie gradientu w celu uniknięcia akumulacji gradientu z poprzedniej iteracji 
            outputs = model(images) # Obliczenie predykcji modelu
            loss = criterion(outputs, labels) # Obliczenie funkcji straty na podstawie predykcji i prawdziwych etykiet 
            
            # Optimizacja modelu
            loss.backward() # Obliczenie gradientu funkcji straty 
            optimizer.step() # Aktualizacja wag modelu na podstawie gradientu funkcji straty 
        
        # Wypisanie funkcji straty dla każdej epoki
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
train_model(model, data_loader, criterion, optimizer)

# Ewaluacja modelu
def evaluate_model(model, data_loader):
    model.eval() # Ustawienie modelu w tryb ewaluacji
    correct = 0 # Liczba poprawnych predykcji
    total = 0 # Liczba wszystkich predykcji
    with torch.no_grad(): # Wyłączenie obliczania gradientu w celu oszczędności pamięci
        for images, labels in data_loader:
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1) # Wybranie klasy z najwyższym prawdopodobieństwem
            total += labels.size(0) # Zwiększenie liczby wszystkich predykcji o rozmiar batcha
            correct += (predicted == labels).sum().item() # Zwiększenie liczby poprawnych predykcji o liczbę poprawnych predykcji w batchu
    print(f'Accuracy: {correct / total * 100:.2f}%')
    
evaluate_model(model, data_loader)