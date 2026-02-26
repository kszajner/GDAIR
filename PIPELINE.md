# PIPELINE ML — GDAIR (Gdańsk Air Inference)

> Uzupełnij sekcje oznaczone **[TY: ...]** zanim uruchomisz agentów.
> Każdy agent czyta swoją sekcję + cały CONTEXT.md przed rozpoczęciem.

---

## KONFIGURACJA PROJEKTU

```
Nazwa projektu:   GDAIR — klasyfikacja ryzyka wysokiego zanieczyszczenia powietrza w Gdańsku

Pliki wejściowe:
  Surowe dane dzienne:          raw_data.csv
  Historia prognoz modelu:      prediction_log.csv
  Stary model:                  rf_model.joblib

WAŻNE — co model robi (przeczytaj uważnie):
  Model NIE prognozuje wartości PM10 ani PM2.5.
  Model klasyfikuje RYZYKO: czy następnego dnia wystąpi wysokie zanieczyszczenie.
  Input:  dane meteorologiczne + aktualne PM10/PM2.5 (jako features opisujące dziś)
  Output: prawdopodobieństwo że jutro PM10 przekroczy zdefiniowany próg
  PM10 i PM2.5 są FEATURES (opis bieżącej sytuacji), nie targetem.

Definicja "wysokiego zanieczyszczenia" (target = 1):
  [TY: sprawdź w model.ipynb — np. PM10 > 50 µg/m³ lub PM10 > 15 µg/m³ (WHO)]
  Progi ryzyka w output: p < 0.30 → low | 0.30–0.50 → moderate | ≥ 0.50 → high

Typ problemu:     klasyfikacja binarna (predict_proba → p(wysokie zanieczyszczenie jutro))

Preferencja błędów:
  WAŻNE: lepiej fałszywy alarm (false positive) niż przeoczenie zagrożenia (false negative).
  Optymalizuj pod RECALL dla klasy "high" — nie chcemy przeoczyć żadnego realnego zagrożenia.
  Metryka priorytetowa: Recall dla klasy 1 (high) ≥ 0.80
  Cel ogólny: accuracy ≥ 0.80 na danych testowych

Kontekst biznesowy:
  Codziennie ~19:00 (Europe/Warsaw) pipeline pobiera dane pogodowe i PM,
  buduje features z okna 3-dniowego i prognozuje ryzyko na jutro.
  Wynik trafia na Discord jako powiadomienie dla użytkowników w Gdańsku.
  Fałszywy alarm (powiedzieliśmy "high" a było OK) = mała uciążliwość dla użytkownika.
  Przeoczenie (powiedzieliśmy "low" a było wysokie) = realny problem zdrowotny.

Źródła danych:
  Pogoda: Open-Meteo API (lat 54.3523, lon 18.6466, tz: Europe/Warsaw)
  PM:     GIOŚ/PJP API, sensory 4706 i 27667
  
Dane w raw_data.csv (jedna obserwacja = jeden dzień):
  date:               data (Europe/Warsaw)
  avg_temperature:    średnia temperatura [°C]
  avg_humidity:       średnia wilgotność [%]
  avg_pressure:       średnie ciśnienie [hPa]
  avg_wind_speed:     średnia prędkość wiatru [m/s]
  sum_precipitation:  suma opadów [mm]
  PM10:               średnie PM10 danego dnia [µg/m³] — to FEATURE, nie target
  PM2_5:              średnie PM2.5 danego dnia [µg/m³] — to FEATURE, nie target
  timestamp:          czas dodania wiersza do CSV (metadana, nie używać jako feature)

Zakres dat:
  [TY: pierwsza i ostatnia data w raw_data.csv]
  Dane treningowe starego modelu: do końca 2024
  Dane produkcyjne (nowe):        od 2025-01-01

Znane problemy ze starym modelem:
  Ogólnie słaba jakość prognoz od początku 2025.
  [TY: opisz co widzisz — np. "zawsze daje low", "za dużo alarmów", itp.]

Metryki sukcesu — PRIORYTETY:
  1. Recall dla klasy "high" ≥ 0.80  ← priorytet nr 1
  2. Accuracy ogólna ≥ 0.80
  3. Precision dla klasy "high" — ważna ale drugoplanowa
  Lepiej mieć Recall=0.90, Precision=0.60 niż Recall=0.65, Precision=0.85

Preferencje techniczne:
  Biblioteki ML:        scikit-learn, xgboost, lightgbm
  Format zapisu:        joblib (wymagane przez predict.py)
  Język raportu:        polski
```

---

## AGENT 1 — Weryfikacja, zrozumienie i feedback

### Rola tego agenta
Jesteś analitykiem który musi **dokładnie zrozumieć projekt** zanim cokolwiek się zmieni.
Twoim jedynym zadaniem jest analiza i przekazanie rzetelnego feedbacku do Agent 2.
Nie trenujesz modeli. Nie modyfikujesz danych.

### Przeczytaj najpierw
- Cały PIPELINE.md (szczególnie sekcję "Konfiguracja projektu")
- `raw_data.csv`
- `prediction_log.csv`
- `model.ipynb` — zrozum jak był trenowany stary model
- `data_preparation.ipynb` — jak były budowane features
- `predict.py` — jak wygląda pipeline produkcyjny
- `CONTEXT.md` sekcja "Twoje uwagi przed startem"

### Zadania

**1. Weryfikacja modelu i danych**

Zrozum co robi stary model:
- Jaki jest dokładny próg PM10 używany do tworzenia targetu? (sprawdź w model.ipynb)
- Jakie features wchodzą do modelu? (sprawdź feature_names z rf_model.joblib)
- Jakie hiperparametry miał RandomForest?
- Czy target był tworzony poprawnie (PM10 z NASTĘPNEGO dnia)?

Sprawdź dane:
- Zakres dat w raw_data.csv
- Braki danych per kolumna i per rok
- Ile dni w danych miało target=1 (wysokie zanieczyszczenie)?
- Proporcja target=1 vs target=0 — czy klasy są zbilansowane?
- Sezonowość — kiedy najczęściej występuje wysokie zanieczyszczenie?
- Luki w datach (brakujące dni)

**2. Analiza błędów starego modelu**

Połącz prediction_log.csv z raw_data.csv po dacie.
Oblicz rzeczywiste targety za okres prognoz (używając tego samego progu co stary model).

Oblicz:
- Accuracy, Recall, Precision dla każdej klasy
- Ile razy model powiedział "low" a było "high"? (najgroźniejszy błąd)
- Ile razy model powiedział "high" a było "low"? (fałszywy alarm)
- Czy błędy są sezonowe?
- Czy błędy korelują z konkretnymi warunkami meteo?

**3. Diagnoza**

Na podstawie analizy oceń:
- Czy model systematycznie zaniża ryzyko (za dużo "low")?
- Czy problemem jest data drift, concept drift, złe hiperparametry, czy coś innego?
- Czy okno 3-dniowe jest właściwe?
- Czy próg klasyfikacji 0.30/0.50 jest optymalny biorąc pod uwagę preferencję Recall?

**4. Wykresy diagnostyczne**

Zapisz do `reports/figures/diagnosis/`:
- Rozkład PM10 w czasie (przed/po 2025)
- Sezonowość PM10 (średnia per miesiąc)
- Prognozowane prawdopodobieństwo vs rzeczywisty target w czasie
- Rozkład błędów (kiedy model mylił się w górę vs w dół)

### Output — FEEDBACK dla Agent 2

Najważniejszy output tej sesji: wypełnij `CONTEXT.md` sekcja Agent 1.
Musi zawierać konkretne, actionable wskazówki dla Agent 2:
- Dokładny próg PM10 do tworzenia targetu
- Czy i jak obsłużyć niezbilansowane klasy
- Strategia danych (wszystkie / tylko 2025 / sample weighting)
- Czy rozszerzyć okno czasowe
- Rekomendowane podejścia do modelowania
- Sugerowane progi dla predict_proba (zamiast 0.30/0.50) które poprawią Recall

### Czego NIE rób
- Nie modyfikuj żadnych plików danych
- Nie trenuj modeli
- Nie przepisuj predict.py

---

## AGENT 2 — Retrenowanie modelu

### Rola tego agenta
Masz przeuczyć model i osiągnąć **accuracy ≥ 0.80** na danych testowych,
przy czym **Recall dla klasy "high" ≥ 0.80** jest ważniejszy niż ogólna accuracy.
Lepiej fałszywy alarm niż przeoczenie zagrożenia.

### Przeczytaj najpierw
- **Cały CONTEXT.md** — szczególnie feedback Agent 1, to Twoja mapa drogowa
- PIPELINE.md sekcja "Konfiguracja projektu"
- `raw_data.csv`, `data_preparation.ipynb`, `model.ipynb`

### Zadania

**1. Przygotowanie danych**

Czyszczenie:
- Usuń kolumnę `timestamp` (metadana)
- Usuń duplikaty dat
- Interpolacja liniowa dla krótkich luk (1-2 dni)
- Zachowaj kolumnę `date` do time-based split

Target — UWAGA NA LEAKAGE:
- Użyj progu PM10 wskazanego przez Agent 1
- Target = 1 jeśli PM10 NASTĘPNEGO dnia > próg
- Przesuń PM10 o 1 dzień do przodu — sprawdź dwukrotnie żeby nie było leakage
- Ostatni wiersz (brak następnego dnia) — usuń ze zbioru treningowego

Feature engineering (okno 3-dniowe + rozszerzenia sugerowane przez Agent 1):
- Lagi: _2, _3 dla wszystkich zmiennych meteo i PM
- Statystyki PM: PM10_avg, PM10_std, PM10_CV, PM2.5_avg, PM2.5_std, PM2.5_CV
- Trendy: WindSpeed_trend, Humidity_diff, PM10_trend, PM2.5_trend
- Kalendarz: Month, IsWeekend, IsHoliday (biblioteka `holidays`, country='PL')
- Dodatkowe features jeśli Agent 1 sugerował (np. szersze okno, rolling mean)

Podział — TYLKO time-based split:
- Train: dane do 3 miesiące przed końcem datasetu
- Test: ostatnie 3 miesiące
- NIE używaj random split

**2. Strategia obsługi niezbilansowanych klas**

Klasy prawdopodobnie są niezbilansowane (mniej dni z wysokim PM10).
Przetestuj kombinacje:
- class_weight='balanced' w modelu
- Sample weighting (nowsze dane ważniejsze) jeśli Agent 1 rekomendował
- SMOTE lub inne oversampling (opcjonalnie, jeśli duży imbalance)

**3. Trenowanie modeli — walidacja TimeSeriesSplit**

Przetestuj minimum 3 podejścia:

Podejście A — RandomForest z tuningiem:
- class_weight='balanced'
- Random search 30 iteracji, TimeSeriesSplit n_splits=5
- Optymalizuj pod Recall klasy 1

Podejście B — Gradient Boosting (XGBoost lub LightGBM):
- scale_pos_weight = (liczba_0 / liczba_1)
- Early stopping
- TimeSeriesSplit CV

Podejście C — Threshold tuning na najlepszym modelu:
- Weź najlepszy model z A lub B
- Przesuń próg decyzyjny poniżej 0.50 żeby zwiększyć Recall
- Znajdź próg gdzie Recall_high ≥ 0.80 przy możliwie wysokiej Precision

**4. Wybór finalnego modelu**

Kryterium wyboru (w kolejności priorytetu):
1. Recall dla klasy "high" ≥ 0.80
2. Accuracy ogólna ≥ 0.80
3. Możliwie wysoka Precision (minimalizacja fałszywych alarmów)

Jeśli żadne podejście nie osiąga obu progów jednocześnie:
- Priorytet ma Recall ≥ 0.80
- Opisz w CONTEXT.md co udało się osiągnąć i dlaczego

**5. Zapis**

- `rf_model_v2.joblib` — nowy model (NIE nadpisuj rf_model.joblib!)
- `data/clean/feature_names.txt` — lista features w kolejności (kluczowe dla predict.py)
- `models/all_results.json` — wyniki wszystkich podejść
- `models/final_metrics.json` — metryki finalnego modelu na test set
- `models/optimal_threshold.json` — optymalny próg decyzyjny + uzasadnienie

**6. Zaktualizowany predict.py**

Zmodyfikuj `predict.py` (backup jako `predict_v1.py`):
- Wczytuje `rf_model_v2.joblib`
- Feature engineering zgodny z tym co zrobiłeś powyżej
- Używa optymalnego progu z `models/optimal_threshold.json`
- Waliduje kolejność features z `data/clean/feature_names.txt`
- Zachowuje format zapisu do `prediction_log.csv`
- Zachowuje kompatybilność z GitHub Actions

**7. Raport** (`reports/report.html`)

Napisz raport po polsku zawierający:
- Podsumowanie diagnozy od Agent 1
- Co zrobiono w retrenowaniu i dlaczego
- Porównanie podejść (tabela z metrykami)
- Wyniki finalnego modelu vs stary model
- Feature importance: stary vs nowy
- Kalibracja prawdopodobieństw
- Nowe progi decyzyjne i ich uzasadnienie
- Ograniczenia i kiedy model może się zdegradować

Wykresy (`reports/figures/`):
- Porównanie metryk: stary vs nowy model
- Confusion matrix finalnego modelu
- ROC curve z zaznaczonym optymalnym progiem
- Feature importance (top 15)
- Prognozowane p(high) w czasie: stary vs nowy model

### Czego NIE rób
- Nie nadpisuj `rf_model.joblib`
- Nie używaj random split — tylko TimeSeriesSplit
- Nie używaj PM10/PM2.5 z tego samego dnia co target jako features bez przesunięcia
- Nie rób grid search — tylko random search max 30 iteracji

---

## Struktura folderów

```
GDAIR/
├── PIPELINE.md              ← ten plik
├── CONTEXT.md               ← historia decyzji
├── START.md                 ← instrukcja uruchamiania
├── raw_data.csv             ← surowe dane (nie modyfikuj!)
├── prediction_log.csv       ← historia prognoz (nie modyfikuj!)
├── rf_model.joblib          ← stary model (nie nadpisuj!)
├── rf_model_v2.joblib       ← nowy model (output Agent 2)
├── predict.py               ← zaktualizowany skrypt produkcyjny
├── predict_v1.py            ← backup
├── data/
│   └── clean/               ← dane przygotowane przez Agent 2
├── models/                  ← metryki i artefakty
└── reports/
    └── figures/
        └── diagnosis/       ← wykresy Agent 1
```
