import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Klasa CustomDataset służy do efektywnego wczytywania i przetwarzania zbioru danych 
    zawierającego obrazy i odpowiadające im etykiety, specyficzne dla potrzeb projektu. 
    Ta klasa jest kluczowa dla projektu, ponieważ standardowe klasy Dataset w Pytorch 
    nie są dostosowane do specyficznych formatów i wymagań, takich jak obsługa plików JSON 
    z etykietami i przetwarzanie obrazów w określony sposób. CustomDataset umożliwia 
    dostosowanie procesu wczytywania danych, co jest istotne dla efektywnego trenowania modeli 
    w Pytorch oraz dla późniejszej analizy i klasyfikacji obrazów.

    Parametry:
    - images_dir: ścieżka do folderu z obrazami.
    - labels_dir: ścieżka do folderu z etykietami (pliki JSON).
    - transform: zestaw transformacji do zastosowania na obrazach (np. zmiana rozmiaru, normalizacja).
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # Zwraca pojedynczy obraz i jego etykietę na podstawie indeksu 'i'
        
        img_name = self.images[i]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Konwersja obrazu do formatu RGB w celu zachowania spójności

        label_name = img_name.replace('.jpg', '.json')
        label_path = os.path.join(self.labels_dir, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label_data = json.load(file)
                label = self.process_label(label_data)
        else:
            label = torch.zeros((16, 4))  # Jeśli nie ma etykiety, wypełnia zerami

        # Przekształcenia obrazu
        if self.transform:
            image = self.transform(image)
        
        return image, label
            
    def process_label(self, label_data):
        """
        Przetwarza dane etykiety na tensor PyTorch. Zakłada, że etykiety są w formacie
        listy słowników, gdzie każdy słownik zawiera klucz 'wspolrzedne' z listą wartości.
        Jako że tensory muszą mieć stały rozmiar, ustawiliśmy go na 16 - tyle najwięcej obiektów jest w jednej z etykiet (zdjecie_15.json)
        Jeśli etykieta ma mniej niż 16 obiektów, to zostanie uzupełniona zerami.

        Zwraca Tensor PyTorch zawierający współrzędne.
        """
        max_objects = 16
        # Inicjalizacja tensora wypełnionego zerami o rozmiarze [16, 4]
        padded_label = torch.zeros((max_objects, 4))

        # Jeśli etykieta zawiera jakiekolwiek dane
        if label_data:
            # Wypełnianie tensora rzeczywistymi współrzędnymi
            for i, label in enumerate(label_data):
                if i >= max_objects:
                    break  # Ograniczenie do maksymalnie 16 obiektów
                padded_label[i] = torch.tensor(label['wspolrzedne'][:4])

        return padded_label




transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # Konwersja obrazu PIL na tensor Pytorch
    transforms.ToTensor(),
    # Normalizacja obrazu (wartości średnie i odchylenia standardowe dla kanałów RGB)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(images_dir='data/obrazy', labels_dir='data/etykiety', transform=transform)
