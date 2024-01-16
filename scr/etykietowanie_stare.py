# Nasz własny program do etykietowania stworzony w Pythonie z wykorzystaniem biblioteki Tkinter
# Biblioteka Tkinter została użyta do stworzenia interfejsu graficznego użytkownika (GUI)
# Biblioteka PIL (Python Imaging Library) została użyta do wczytywania obrazów
# Program ten pozwala na wczytanie wielu obrazów naraz, zachowując ich kolejność i informuje użytkownika o tym, które zdjęcie jest obecnie wyświetlane i ile zostało
# Użytkownik może zaznaczyć prostokąty na obrazie, które reprezentują dany obiekt. Program potwierdza w konsoli, że prostokąt został dodany i zapisuje współrzędne prostokąta w liście
# Gdy użytkownik przejdzie do następnego zdjęcia, program zapisuje listę współrzędnych do pliku JSON o tej samej nazwie co zdjęcie, ale z rozszerzeniem .json
# Program został napisany, gdy myśleliśmy, że chodzi o detekcję obiektów, a nie klasyfikację, więc nie jest on już potrzebny, ale zostawiam go tutaj, gdyby ktoś chciał go użyć w przyszłości

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json

class AplikacjaEtykietowania:
    def __init__(self, root):
        self.root = root
        self.root.title("Narzędzie do Etykietowania Samochodów")

        self.MAX_IMAGE_SIZE = (1400, 1000) # Maksymalny rozmiar obrazu, można zmienić zależnie od preferencji użytkownika

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        self.obrazy = []
        self.obecny_obraz_idx = 0
        self.obecny_obraz = None
        self.obecny_obrazTk = None
        self.prostokat = None
        self.start_x = None
        self.start_y = None
        self.etykiety = []

        self.setup_gui()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def setup_gui(self):
        self.licznik_zdjec_label = tk.Label(self.root, text="Zdjęcie: 0 z 0")
        self.licznik_zdjec_label.pack(side=tk.TOP, fill=tk.X)

        przycisk_otworz = tk.Button(self.root, text="Otwórz Zdjęcia", command=self.otworz_zdjecia, bg='lightblue', fg='black')
        przycisk_otworz.pack(side=tk.TOP, fill=tk.X)

        przycisk_nastepny = tk.Button(self.root, text="Następne Zdjęcie", command=self.nastepne_zdjecie, bg='lightgreen', fg='black')
        przycisk_nastepny.pack(side=tk.TOP, fill=tk.X)

    def otworz_zdjecia(self):
        sciezki_plikow = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if sciezki_plikow:
            self.obrazy = list(sciezki_plikow)
            self.obecny_obraz_idx = 0
            self.wczytaj_obraz()

    def wczytaj_obraz(self):
        if self.obecny_obraz_idx < len(self.obrazy):
            self.aktualizuj_licznik_zdjec()
            self.obecny_obraz = Image.open(self.obrazy[self.obecny_obraz_idx])
            self.obecny_obraz.thumbnail(self.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)  
            self.obecny_obrazTk = ImageTk.PhotoImage(self.obecny_obraz)
            self.canvas.config(width=self.obecny_obrazTk.width(), height=self.obecny_obrazTk.height())
            self.canvas.create_image(0, 0, anchor="nw", image=self.obecny_obrazTk)
            self.etykiety.clear()
        else:
            messagebox.showinfo("Koniec", "Wszystkie zdjęcia zostały zaetykietowane.")


    def nastepne_zdjecie(self):
        self.zapisz_etykietowanie()
        self.obecny_obraz_idx += 1
        self.wczytaj_obraz()

    def aktualizuj_licznik_zdjec(self):
        licznik_tekst = f"Zdjęcie: {self.obecny_obraz_idx + 1} z {len(self.obrazy)}"
        self.licznik_zdjec_label.config(text=licznik_tekst)


    def zapisz_etykietowanie(self):
        if self.etykiety:
            nazwa_pliku = os.path.basename(self.obrazy[self.obecny_obraz_idx])
            nazwa_pliku_bez_rozszerzenia = os.path.splitext(nazwa_pliku)[0]
            folder_etykiet = '../data/etykiety'
            if not os.path.exists(folder_etykiet):
                os.makedirs(folder_etykiet)
            sciezka_pliku = os.path.join(folder_etykiet, f"{nazwa_pliku_bez_rozszerzenia}.json")
            with open(sciezka_pliku, "w") as f:
                json.dump(self.etykiety, f)
            print(f"Etykiety zapisane do {sciezka_pliku}")
            self.etykiety.clear()
        else:
            print("Brak etykiet do zapisania.")

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.prostokat = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        curX, curY = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.prostokat, self.start_x, self.start_y, curX, curY)

    def on_release(self, event):
        if self.prostokat:
            x1, y1, x2, y2 = self.canvas.coords(self.prostokat)
            self.etykiety.append((x1, y1, x2, y2))
            print(f"Dodano etykietę: {x1, y1, x2, y2}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikacjaEtykietowania(root)
    root.mainloop()