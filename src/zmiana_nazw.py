import os

def zmien_nazwy_zdjec(folder, nowa_nazwa_podstawowa):
    """
    Zmienia nazwy zdjęć w podanym folderze na kolejne numery (1, 2, 3, ..., 300).

    folder (str): Ścieżka do folderu ze zdjęciami.
    nowa_nazwa_podstawowa (str): Podstawowa część nowej nazwy pliku (np. 'zdjecie').
    """
    sciezki_plikow = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, sciezka in enumerate(sciezki_plikow, start=1):
        nowa_nazwa = f"{nowa_nazwa_podstawowa}_{i}.jpg"
        nowa_sciezka = os.path.join(folder, nowa_nazwa)
        os.rename(sciezka, nowa_sciezka)
        print(f"Zmieniono nazwę: {sciezka} -> {nowa_sciezka}")

    print("Zmiana nazw zakończona.")

# Użycie skryptu
folder_zdjec = '../data/selected_images'  # Zastąp odpowiednią ścieżką
zmien_nazwy_zdjec(folder_zdjec, "zdjecie")
