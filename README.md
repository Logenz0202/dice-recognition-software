# Dice Recognition Software (working title) – Rozpoznawanie kości RPG i ich wyników

Celem aplikacji jest automatyczne rozpoznawanie rodzaju kości wielościennych (D4, D6, D8, D10, D12, D20) oraz wartości wyrzuconych na ich ściankach, przy użyciu algorytmów rozpoznawania obrazu i uczenia maszynowego.

---

## Technologie i założenia

- Rozpoznawanie:
  - typu kości na podstawie kształtu (ilość ścian)
  - wartości wyrzuconej (liczby na ściance)
- Użycie **konwolucyjnej sieci neuronowej (CNN)** trenowanej na niestandardowym zbiorze zdjęć.
- Dane treningowe:
  - Zdjęcia z 5 różnych zestawów kości (różne materiały i kolory)
  - Augmentacja (obrót, rozmycie, kontrast, jasność, szum itd.)
- Możliwość integracji z aplikacjami RPG jako inteligentny licznik rzutów.

---

## Struktura projektu i zbioru danych

Każde oryginalne zdjęcie ma nazwę w formacie:

```
[dice_type]_[face_value]_[set_id]_[img_id].jpg
```

**Przykład:**  
`d20_17_set3_004.jpg` – To zdjęcie przedstawia kość D20 z wynikiem 17 z zestawu nr 3.

Dodatkowo, **dla każdego zdjęcia wygenerowano 10 wersji poprzez augmentację**, których nazwy zawierają sufiks:

```
..._aug[1-10].jpg
```

**Przykład:**  
`d20_17_set3_004_aug3.jpg` – Trzeci wariant wspomnianego wcześniej zdjęcia.

---

## Wymagania

- Python 3.8+
- TensorFlow lub PyTorch
- OpenCV lub Pillow
- Biblioteki do augmentacji: `albumentations` lub `imgaug`

---

## Etapy projektu

- [X] Zebranie i oznaczenie danych (obrazy kości różnych typów i zestawów)
- [X] Augmentacja danych
- [X] Rozpoznawanie liczby (klasyfikacja cyfr)
- [X] Rozpoznawanie typu kości (klasyfikacja kształtu)
- [X] Budowa i trenowanie końcowego modelu CNN
- [X] Interfejs graficzny lub CLI (opcjonalnie)
- [X] Ewaluacja dokładności i testy krzyżowe

---

## Dalsze plany

- Rozpoznawanie **wielu kości jednocześnie** na jednym zdjęciu
- Umożliwienie użytkownikowi wykonywania operacji na wynikach, takich jak:
  - **Sumowanie wartości** ze wszystkich wyrzuconych kości lub tylko konkretnego typu
  - **Wybieranie najwyższego/najniższego wyniku** z rzutu
- Dodanie opcji **filtrowania wyników według typu kości**
- Wprowadzenie **interfejsu API lub integracji z aplikacjami RPG online**, umożliwiającego np. przesyłanie wyników do Roll20, Foundry VTT itp.

---

## Autorzy

Projekt realizowany w ramach przedmiotu **Inteligencja Obliczeniowa** przez:
- Szymon Ligenza, GitHub: [Logenz0202](https://github.com/Logenz0202)
- Bartosz Stromski, GitHub: [Sacharow](https://github.com/Sacharow)
