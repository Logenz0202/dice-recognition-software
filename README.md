# Dice Recognition Software (working title) – Rozpoznawanie kości RPG i ich wyników

Projekt na przedmiot **Inteligencja Obliczeniowa**.

Celem aplikacji jest automatyczne rozpoznawanie rodzaju kości wielościennych (D4, D6, D8, D10, D12, D20) oraz wartości wyrzuconych na ich ściankach, przy użyciu algorytmów rozpoznawania obrazu i uczenia maszynowego.

---

## Technologie i założenia

- Rozpoznawanie:
  - typu kości na podstawie kształtu (ilość ścian)
  - wartości wyrzuconej (liczby na ściance)
- Użycie **konwolucyjnej sieci neuronowej (CNN)** trenowanej na niestandardowym zbiorze zdjęć.
- Dane treningowe:
  - Zdjęcia z 5 różnych zestawów kości (różne czcionki, materiały, kolory)
  - Augmentacja (obrót, rozmycie, kontrast, jasność, szum itd.)
- Możliwość integracji z aplikacjami RPG jako inteligentny licznik rzutów.

---

## Struktura zbioru danych

Każdy plik ma nazwę w formacie:

```
[dice_type]_[face_value]_[set_id]_[img_id].jpg
```

**Przykład:** `d20_17_set3_004.jpg`  
To zdjęcie przedstawia kość D20 z wynikiem 17 z zestawu nr 3.

---

## Wymagania

- Python 3.8+
- TensorFlow lub PyTorch
- OpenCV lub Pillow
- Biblioteki do augmentacji: `albumentations` lub `imgaug`

---

## Etapy projektu

- [ ] Zebranie i oznaczenie danych (obrazy kości różnych typów i zestawów)
- [ ] Augmentacja danych
- [ ] Rozpoznawanie liczby (klasyfikacja cyfr)
- [ ] Rozpoznawanie typu kości (klasyfikacja kształtu)
- [ ] Budowa i trenowanie końcowego modelu CNN
- [ ] Interfejs graficzny lub CLI (opcjonalnie)
- [ ] Ewaluacja dokładności i testy krzyżowe

---

## Autorzy

Projekt realizowany jako część kursu **Inteligencja Obliczeniowa** przez:
- Szymon Ligenza, github: [Logenz0202](https://github.com/Logenz0202)
- Bartosz Stromski, github: [Sacharow](https://github.com/Sacharow)
