# Lab 11 TODO

## 0. Przygotowanie
- [ ] Rozpakuj dane z `lab11/data/*.zip` do katalogów roboczych.
- [ ] Sprawdź, czy w `lab11/data/pairs` są poprawne pary `left_XX` i `right_XX`.
- [ ] Ustaw parametry planszy kalibracyjnej (liczba pól wewnętrznych, rozmiar pola).

## 1. Ćwiczenie 11.1 — Kalibracja pojedynczej kamery
- [ ] Wykryj narożniki szachownicy dla obrazów lewej kamery.
- [ ] Wykryj narożniki szachownicy dla obrazów prawej kamery.
- [ ] Wyznacz parametry kamery (`K`, `D`) osobno dla lewej i prawej.
- [ ] Wyświetl parametry oraz błąd RMS.
- [ ] Wykonaj korekcję zniekształceń dla przykładowego obrazu.
- [ ] Porównaj obraz przed i po korekcji (test prostych linii).

## 2. Ćwiczenie 11.2 — Kalibracja i rektyfikacja stereo
- [ ] Zbuduj wspólny zbiór punktów dla par, gdzie wykryto narożniki po obu stronach.
- [ ] Wykonaj kalibrację stereo (`stereoCalibrate`).
- [ ] Wyznacz rektyfikację (`stereoRectify`) i mapy przekształceń (`initUndistortRectifyMap`).
- [ ] Zastosuj `remap` dla przykładowej pary obrazów.
- [ ] Połącz obrazy i narysuj poziome linie kontrolne.
- [ ] Potwierdź, że punkty odpowiadające leżą w tych samych wierszach.

## 3. Ćwiczenie 11.3 — Mapa dysparycji (BM i SGBM)
- [ ] Policz mapę dysparycji BM dla obrazu przed rektyfikacją.
- [ ] Policz mapę dysparycji BM dla obrazu po rektyfikacji.
- [ ] Policz mapę dysparycji SGBM dla obrazu przed rektyfikacją.
- [ ] Policz mapę dysparycji SGBM dla obrazu po rektyfikacji.
- [ ] Znormalizuj mapy do zakresu 0–255.
- [ ] Dodaj heatmapy (`cv2.applyColorMap`) i porównanie wyników.
- [ ] Dostrań parametry (`numDisparities`, `blockSize`, `P1`, `P2`, `uniquenessRatio`).

## 4. Walidacja i domknięcie
- [ ] Sprawdź, czy notebook wykonuje się od początku do końca.
- [ ] Upewnij się, że kod jest funkcyjny (bez duplikacji logiki).
- [ ] Upewnij się, że wywołania własnych funkcji używają argumentów nazwanych.
- [ ] Sprawdź ostrzeżenia/błędy narzędzi statycznych.
- [ ] Uzupełnij krótkie wnioski z porównania BM vs SGBM.
