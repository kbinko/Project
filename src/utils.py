from PIL import Image
import numpy as np

def load_image(image_path):
    """
    Pobiera obraz z podanej ścieżki i zwraca jego kopię.
    """
    try:
        img = Image.open(image_path)
        img_copy = img.copy()  # Tworzenie kopii obrazu
        img.close()  # Zamknięcie oryginalnego obrazu
        return img_copy
    except IOError:
        print(f"Błąd podczas ładowania obrazu: {image_path}")
        return None

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
