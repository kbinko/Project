import json
import os

def napraw_etykiety(folder_etykiet):
    for nazwa_pliku in os.listdir(folder_etykiet):
        if nazwa_pliku.endswith(".json"):
            sciezka_pliku = os.path.join(folder_etykiet, nazwa_pliku)

            with open(sciezka_pliku, 'r') as plik:
                dane = json.load(plik)
                poprawione_dane = []

                for etykieta in dane:
                    # Sprawdzenie, czy etykieta jest zagnieżdżona
                    if isinstance(etykieta['wspolrzedne'], list):
                        poprawione_dane.append(etykieta)
                    else:
                        # Rozwijanie zagnieżdżonych etykiet
                        aktualna_etykieta = etykieta
                        while not isinstance(aktualna_etykieta['wspolrzedne'], list):
                            aktualna_etykieta = aktualna_etykieta['wspolrzedne']
                        poprawione_dane.append({"nazwa": "samochod", "wspolrzedne": aktualna_etykieta['wspolrzedne']})

            # Zapisanie poprawionych danych
            with open(sciezka_pliku, 'w') as plik:
                json.dump(poprawione_dane, plik)

# Ścieżka do folderu z etykietami
folder_etykiet = '../data/etykiety'  # Zmień na właściwą ścieżkę
napraw_etykiety(folder_etykiet)
