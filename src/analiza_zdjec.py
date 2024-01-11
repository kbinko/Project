import os
import shutil
from utils import load_image, ocena_jakosci

def analiza(source_folder, destination_folder, top_n=300):
    """
    Analizuje zdjęcia w podanym folderze, sortuje je według jakości i wybiera top_n zdjęć.
    """
    zdjecia = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    wyniki_jakosci = {}

    for i, img_path in enumerate(zdjecia):
        print(f"Przetwarzanie obrazu {i + 1} z {len(zdjecia)}")
        image = load_image(img_path)
        if image:
            quality_metrics = ocena_jakosci(image)
            # Ocena jakości na podstawie zdefiniowanych metryk
            wynik = quality_metrics["resolution"][0] * quality_metrics["resolution"][1] + quality_metrics["sharpness"]
            wyniki_jakosci[img_path] = wynik

    # Sortowanie zdjęć według oceny jakości i wybór top_n
    top_images = sorted(wyniki_jakosci, key=wyniki_jakosci.get, reverse=True)[:top_n]

    # Kopiowanie najlepszych zdjęć do docelowego folderu
    for img in top_images:
        shutil.copy(img, os.path.join(destination_folder, os.path.basename(img)))

    print(f"Skopiowano {len(top_images)} zdjęć do {destination_folder}")
    return top_images

def main():
    source_folder = '../data/input_images'  # Ścieżka do folderu ze zdjęciami do analizy
    destination_folder = '../data/selected_images'  # Ścieżka do folderu docelowego

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    top_images = analiza(source_folder, destination_folder)

if __name__ == "__main__":
    main()
