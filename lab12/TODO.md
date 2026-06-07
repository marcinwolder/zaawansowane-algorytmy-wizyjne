# Lab 12: Termowizja — TODO

## 0. Przygotowanie notebooka (`lab12/lab12.ipynb`)

- [x] Utwórz/uzupełnij notebook `lab12/lab12.ipynb`.
- [x] Ustal strukturę sekcji:
  - [x] `12.1 Prosta analiza obrazu termowizyjnego`
  - [x] `12.2 YOLOv3: RGB + termowizja (early/late fusion)`
  - [x] `12.3 Wzorzec probabilistyczny (nieobowiązkowe)`
- [x] Wszystkie komórki **markdown po polsku**.
- [x] Wszystkie komórki **code po angielsku**.
- [x] Zaimplementuj funkcje pomocnicze, aby unikać duplikacji kodu.
- [x] W funkcjach dodaj parametry do ręcznego strojenia (progi, rozmiary filtrów, IoU, itp.).
- [x] Przy wywołaniach własnych funkcji stosuj głównie argumenty nazwane (keyword arguments).

## 1. Ćwiczenie 12.1 — Prosta analiza obrazu termowizyjnego

- [x] Wczytaj sekwencję termowizyjną (np. `lab12/vid1_IR.avi`).
- [x] Zaimplementuj pipeline detekcji:
  - [x] binaryzacja obrazu,
  - [x] filtracja morfologiczna/szumów,
  - [x] etykietowanie obiektów (connected components).
- [x] Odfiltruj komponenty po proporcji `height/width`.
- [x] Połącz wykryte fragmenty sylwetki do jednego bounding boxa na obiekt.
- [x] Narysuj wynikowe ramki na obrazie.
- [x] Dodaj parametry strojenia:
  - [x] próg binaryzacji,
  - [x] typ/rozmiar filtracji,
  - [x] zakres akceptowanych proporcji,
  - [x] warunek łączenia fragmentów.
- [x] Zweryfikuj działanie na wielu klatkach i zapisz przykładowe wyniki.

## 2. Ćwiczenie 12.2 — Detektor YOLOv3 + fuzja RGB/IR

- [x] Przygotuj dane i pliki modelu (zgodnie z materiałami do laboratorium):
  - [x] ścieżki do `test_rgb` i `test_thermal`,
  - [x] pliki wag i konfiguracji YOLOv3.
- [x] Zaimplementuj przełącznik trybu `FUSION`:
  - [x] `EARLY`,
  - [x] `LATE`.

### 2.1 Early fusion

- [x] Wczytaj obraz RGB (OpenCV: BGR) i obraz termowizyjny (grayscale).
- [x] Zbuduj obraz wejściowy dla detektora:
  - [x] kanały `B` i `G` bez zmian,
  - [x] kanał `R = max(R, thermal_gray)`.
- [x] Uruchom detekcję YOLOv3 i zapisz bounding boxy.

### 2.2 Late fusion

- [x] Uruchom osobny detektor dla RGB.
- [x] Uruchom osobny detektor dla termowizji (skala szarości).
- [x] Zaimplementuj IoU dla par ramek (`box_rgb`, `box_ir`).
- [x] Dla wszystkich par o `IoU > 0`:
  - [x] posortuj malejąco po IoU,
  - [x] zachłannie dobierz pary (każda ramka użyta max 1 raz),
  - [x] uśrednij współrzędne i rozmiary,
  - [x] rzutuj wynik na `int`.
- [x] Zwróć finalną listę `boxes`.

### 2.3 Porównanie metod

- [x] Uruchom i porównaj `EARLY` vs `LATE` na tych samych danych.
- [x] Opisz jakościowo wyniki (trafienia, fałszywe alarmy, stabilność ramek).
- [x] (Opcjonalnie) dodaj prosty pomiar czasu inferencji obu podejść.

## 3. Ćwiczenie 12.3 (nieobowiązkowe) — Wzorzec probabilistyczny

### 3.1 Budowa bazy sylwetek

- [x] Dodaj możliwość wycinania ROI sylwetki i zapisu do PNG.
- [ ] Zbierz ręcznie ok. 30–50 próbek sylwetek.
- [ ] Ustandaryzuj rozmiar próbek do `192x64`.

### 3.2 Tworzenie wzorca

- [x] Wczytaj próbki i zbinaryzuj.
- [x] Zsumuj obrazy binarne: `PDM += B`.
- [x] Znormalizuj przez liczbę próbek.
- [x] Zapisz wzorzec do pliku.

### 3.3 Detekcja sliding window

- [x] Wczytaj klatkę testową i wzorzec.
- [x] Wyznacz:
  - [x] `PDM1` (0–1, `float32`),
  - [x] `PDM0 = 1 - PDM1`.
- [x] Zbinaryzuj klatkę i przesuwaj okno `192x64`.
- [x] Dla każdego okna policz miarę dopasowania:
  - [x] `sum(B * PDM1 + (1 - B) * PDM0)`.
- [x] Znormalizuj mapę odpowiedzi do `uint8` dla wizualizacji.
- [x] Wykryj maksimum lokalne i narysuj bounding box.
- [x] (Opcjonalnie) dodaj multi-detekcję: próg + NMS.
- [ ] (Opcjonalnie) opisz lub dodaj detekcję wieloskalową.

## 4. Jakość kodu i walidacja

- [x] Sprawdź notebook pod kątem błędów i ostrzeżeń `ruff`.
- [x] Sprawdź notebook pod kątem ostrzeżeń/błędów Pylance.
- [x] Usuń duplikację kodu przez funkcje.
- [x] Upewnij się, że funkcje mają parametry strojenia.
- [x] Upewnij się, że wywołania własnych funkcji używają keyword arguments.
- [ ] Przejdź notebook od początku do końca (`Run All`) bez błędów.
- [x] Zapisz finalną wersję `lab12/lab12.ipynb`.
