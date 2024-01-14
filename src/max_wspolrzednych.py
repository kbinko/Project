import os
import json

folder_etykiet = '../data/etykiety'  

def max_ilosc_wspolrzednych(folder):
    max = 0
    plik = None
    for nazwa_pliku in os.listdir(folder):
        if nazwa_pliku.endswith(".json"):
            sciezka_pliku = os.path.join(folder, nazwa_pliku)
            with open(sciezka_pliku, 'r') as plik:
                dane = json.load(plik)
                temp = len(dane)
                if temp > max:
                    max = temp
                    plik = nazwa_pliku
    
    print(f'Najwięcej współrzędnych ma plik: {plik} - {max} współrzędnych')

max_ilosc_wspolrzednych(folder_etykiet)
    
