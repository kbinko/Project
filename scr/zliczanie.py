import os
import json

# Ścieżka do katalogu z etykietami
labels_dir = '../data/etykiety'

# Inicjalizacja liczników
count_obecnosc_znaku_1 = 0
count_obecnosc_znaku_0 = 0

# Przeszukiwanie plików JSON w katalogu
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.json'):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as file:
            label_data = json.load(file)
            if label_data.get("obecnosc_znaku") == 1:
                count_obecnosc_znaku_1 += 1
            elif label_data.get("obecnosc_znaku") == 0:
                count_obecnosc_znaku_0 += 1

# Wyświetlenie wyników
print("Liczba etykiet 'obecnosc_znaku: 1':", count_obecnosc_znaku_1)
print("Liczba etykiet 'obecnosc_znaku: 0':", count_obecnosc_znaku_0)
