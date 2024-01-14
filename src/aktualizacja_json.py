import json
import os

def aktualizuj_pliki_json(folder, start, stop):
    for nazwa_pliku in os.listdir(folder):
        if nazwa_pliku.endswith(".json"):
            numer_zdjecia = int(nazwa_pliku.split('_')[1].split('.')[0])
            if start <= numer_zdjecia <= stop:
                sciezka_pliku = os.path.join(folder, nazwa_pliku)

            with open(sciezka_pliku, 'r') as plik:
                dane = json.load(plik)
                zaktualizowane_dane = [{'nazwa': 'samochod', 'wspolrzedne': wspolrzedne} for wspolrzedne in dane]

            with open(sciezka_pliku, 'w') as plik:
                json.dump(zaktualizowane_dane, plik)


folder_etykiet = '../data/etykiety'  # Zastąp ścieżką do folderu zawierającego pliki JSON
aktualizuj_pliki_json(folder_etykiet, 151, 300)
