# Skrypt sprawdzający maksymalną liczbę współrzędnych w plikach z etykietami, zwracający maksymalną liczbę współrzędnych i odpowiadającą nazwą pliku.

import json
import os

def znajdz_maksymalna_liczbe_etykiet(folder_etykiet):
    maksymalna_liczba = 0
    nazwa = ""
    for nazwa_pliku in os.listdir(folder_etykiet):
        if nazwa_pliku.endswith(".json"):
            sciezka_pliku = os.path.join(folder_etykiet, nazwa_pliku)
            with open(sciezka_pliku, 'r') as plik:
                etykiety = json.load(plik)
                liczba_etykiet = len(etykiety)
                if liczba_etykiet > maksymalna_liczba:
                    maksymalna_liczba = liczba_etykiet
                    nazwa = nazwa_pliku
    return maksymalna_liczba, nazwa

# Użyj funkcji do znalezienia maksymalnej liczby etykiet w folderze etykiet
folder_etykiet = '../data/etykiety'  # Zmień na właściwą ścieżkę
maks_etykiet = znajdz_maksymalna_liczbe_etykiet(folder_etykiet)
print("Maksymalna liczba etykiet: ", maks_etykiet[0])
print("Nazwa pliku z maksymalną liczbą etykiet: ", maks_etykiet[1])
