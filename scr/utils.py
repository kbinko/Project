# plik z przydatnymi funkcjami używanymi w innych plikach

import numpy as np
import os
import json
from PIL import Image

def load_image(image_path):
    """
    Funkcja do wczytywania obrazu z dysku.

    Zwraca obiekt obrazu PIL skonwertowany do formatu RGB - ponieważ Pytorch oczekuje obrazów w tym formacie.
    """
    return Image.open(image_path).convert('RGB')

def load_label(label_path):
    """
    Funkcja do wczytywania etykiet z pliku JSON.

    Zwraca dane etykiety w formacie JSON (lista słowników).
    """
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            return json.load(file)
    else:
        return []  # Pusta lista, jeśli plik z etykietami nie istnieje


def ocena_jakosci(image):
    """
    Sprawdza jakość obrazu i zwraca słownik z metrykami.
    """
    quality_metrics = {
        "resolution": image.size,
        "sharpness": ocena_ostrosci(image),
    }
    return quality_metrics

def ocena_ostrosci(image):
    """
    Ocena ostrości obrazu.
    """
    image_array = np.array(image.convert('L'))  # Konwersja do skali szarości
    gradient = np.gradient(image_array)
    sharpness = np.std(gradient)
    return sharpness
