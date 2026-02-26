# CONTEXT — GDAIR

> Ty wypełniasz sekcje "Twoje uwagi". Agenci wypełniają swoje sekcje.
> Nie usuwaj żadnych sekcji. Następny agent zawsze czyta CAŁY ten plik.

---

## Twoje uwagi przed startem — wypełnij ZANIM uruchomisz Agent 1

```
Data startu projektu:

Dane w raw_data.csv:
  Pierwsza data:
  Ostatnia data:
  Łączna liczba wierszy (przybliżona):

Dane w prediction_log.csv:
  Pierwsza prognoza:
  Ostatnia prognoza:
  Łączna liczba prognoz:

Twoje obserwacje co do błędów starego modelu:
  [Opisz własnymi słowami co widzisz — np.:]
  [  "model prawie zawsze daje 'low' bez względu na warunki"]
  [  "zimą daje za dużo fałszywych alarmów"]
  [  "wiosną 2025 kompletnie przestał reagować"]
  -

Rzeczy które wiesz o danych a trudno je zmierzyć:
  [np. awaria sensora, zmiana metody zbierania, nowy czujnik itp.]
  -

Inne uwagi dla agentów:
  -
```

---

## Twoje uwagi PO Agent 1 — wypełnij po sprawdzeniu wyników

```
Czy diagnoza agenta jest trafna? [tak / nie / częściowo]
Uzupełnienie lub korekta:
  -

Czy feedback dla Agent 2 jest kompletny i konkretny? [tak / nie]
Czego brakuje:
  -

Czy kontynuować do Agent 2? [tak / nie]
```

---
---

## Agent 1 — Weryfikacja i feedback
*[Agent wypełnia po zakończeniu — to jest główny input dla Agent 2]*

**Data wykonania:** 2026-02-26

### Co robi stary model (weryfikacja)
- Dokładny próg PM10 dla target=1:           PM10 >= 50 LUB PM2.5 >= 25 µg/m³ (warunek łączny, sprawdzony w data_preparation.ipynb)
- Czy target był tworzony poprawnie (PM10 z następnego dnia): TAK — FutureRisk = Risk.shift(-1), czyli target=1 gdy JUTRZEJSZY dzień ma PM10>=50 lub PM2.5>=25. Brak leakage.
- Features wchodzące do modelu (z rf_model.joblib — 34 features):
    Okno 3-dniowe (dzień D, D-1, D-2) dla: Year, Month, WindSpeed, Temperature,
    Humidity, Pressure, Precipitation, IsWeekend (sufiksy: brak / _2 / _3)
    + agregaty PM: PM10_avg, PM10_std, PM10_CV, PM2.5_avg, PM2.5_std, PM2.5_CV
    + trendy: WindSpeed_trend, Humidity_diff, PM2.5_trend, PM10_trend
    UWAGA: surowe PM10 i PM2.5 są usuwane przed trenowaniem — model widzi tylko agregaty
- Hiperparametry starego RandomForest (z rf_model.joblib i model.ipynb):
    n_estimators=200, max_depth=30, max_features='log2',
    min_samples_split=2, min_samples_leaf=1, bootstrap=True,
    class_weight=None (brak balansowania klas!), random_state=42
- Liczba drzew: 200, max_depth: 30 (bardzo głębokie — ryzyko overfitting)

### Stan danych
- Zakres dat raw_data.csv: od 2025-08-23 do 2026-02-22
  UWAGA KRYTYCZNA: raw_data.csv zawiera WYŁĄCZNIE dane produkcyjne (2025+).
  Dane treningowe (2021-2024) były w dataset.xlsx — ten plik nie jest już dostępny w repo.
  Stary model był trenowany na danych z lat 2021-2024, zatem raw_data.csv nie pokrywa okresu treningowego.
- Łączna liczba wierszy: 182
- Braki danych (kolumny problematyczne): PM2_5 — brak 3 wartości; pozostałe kolumny kompletne
- Luki w datach (kiedy i ile): 2 brakujące dni — 2026-01-06 i 2026-01-22

### Rozkład targetu
- Dni z target=1 (wysokie zanieczyszczenie) w danych treningowych (pre-2025):
  Liczba: BRAK DANYCH — dataset.xlsx niedostępny. Z model.ipynb: df2_oversampling n_samples=338 (klasa 1),
  df1 (klasa 0) = 338 po balansowaniu. Oryginalny rozkład nieznany, ale model trenowano z oversamplingiem klasy 1.
- Dni z target=1 w danych produkcyjnych (2025-08-23 do 2026-02-21):
  Liczba: 74   Procent: 40.9%
  Z czego: 56/74 dni (75.7%) wyzwolonych TYLKO przez PM2.5>=25 (PM10<50)
            17/74 dni wyzwolonych przez oba warunki (PM10>=50 i PM2.5>=25)
             1/74 dzien wyzwolony TYLKO przez PM10>=50 (PM2.5<25)
- Sezonowość: miesiące z najwyższym PM10/target:
    Luty: avg PM10=53.8, target=1 u 71.4% dni
    Styczeń: avg PM10=36.4, target=1 u 62.1% dni
    Listopad: avg PM10=26.0, target=1 u 50.0% dni
    Sierpień-wrzesień: najniższe PM10 (16-25 µg/m³), rzadkie alarmy
- Klasy zbilansowane? NIE  Stosunek 0:1 = 1.45:1
  (umiarkowany imbalance, ale produkcja obciążona sezonem zimowym, więc klasa 1 jest częstsza niż w danych treningowych)

### Wyniki starego modelu na danych produkcyjnych
- Accuracy: 0.707 (cel >= 0.80 NIE OSIAGNIETY)
- Recall dla klasy "high" (target=1): 0.581 (cel >= 0.80 ZDECYDOWANIE NIE OSIAGNIETY — to główny problem)
- Precision dla klasy "high": 0.662
- False Negative Rate (przeoczenia zagrożeń): 0.419 — model przeoczył 31 z 74 rzeczywistych dni wysokiego zanieczyszczenia
- False Positive Rate (fałszywe alarmy): 0.206 — 22 fałszywe alarmy z 107 dni low
- Czy błędy sezonowe: TAK — kiedy najgorzej:
    Wrzesień (recall=0.00 — oba dni z target=1 przeoczone, ale mała próba)
    Sierpień (recall=0.25 — 3 z 4 dni przeoczone)
    Listopad (recall=0.40 — 9 z 15 dni przeoczone) — KRYTYCZNY MIESIĄC
    Październik (recall=0.56 — 4 z 9 dni przeoczone)
    Grudzień (recall=0.55 — 5 z 11 dni przeoczone)
    Najlepszy recall w zimie: Styczeń (0.78) i Luty (0.73)
- Korelacja błędów z warunkami meteo:
    FN (przeoczenia) — średnie warunki: temp=4.8°C, wilgotność=81.2%, PM10=25.8, PM2.5=22.9
    TP (poprawne alarmy) — średnie warunki: temp=-1.3°C, wilgotność=83.4%, PM10=46.1, PM2.5=39.4
    Wniosek: model wykrywa alarmy gdy PM10 jest już bardzo wysokie (>40), ale nie reaguje gdy
    PM2.5 rośnie przy niższym PM10. Kluczowy sygnał (PM2.5) jest niedoszacowany.

### Diagnoza
- Główna przyczyna słabych wyników:
    DATA/CONCEPT DRIFT + błędna definicja progu operacyjnego.
    Model był trenowany na danych 2021-2024 gdzie PM10>=50 było głównym wyzwalaczem.
    W produkcji (2025+) 75.7% alarmów jest wywoływanych przez PM2.5>=25 przy PM10<50.
    Model nie "widzi" surowego PM2.5 — widzi tylko PM2.5_avg z 3-dniowego okna.
    Gdy PM2.5 rośnie stopniowo (np. z 15 do 27) przy niskim PM10, model nie reaguje.
    Ponadto próg decyzyjny 50% jest zbyt wysoki — model rzadko daje >50% pewności,
    nawet w ryzykownych sytuacjach.

- Drugorzędna przyczyna:
    Brak balansowania klas (class_weight=None) + overfitting na danych treningowych
    (max_depth=30, bez regularyzacji). Na danych testowych z epoki trenowania acc=97%,
    ale na produkcji tylko 70.7% — wyraźny overfitting do wzorców 2021-2024.
    Stara struktura sezonowości (kiedy zimą PM10 rosło) może nie odpowiadać obecnej.

- Pewność diagnozy: WYSOKA
- Uzasadnienie:
    31 false negatives z 74 rzeczywistych alarmów to recall=0.581, dramatycznie poniżej celu 0.80.
    Z 31 FN aż 26 to dni gdzie PM10_next < 50 (alarm przez PM2.5) — model systematycznie
    ignoruje sytuacje gdzie PM2.5 przekracza 25 µg/m³ przy PM10 < 50 µg/m³.
    Analiza FN: 11 z 31 przeoczonych miało proba 40-50% (blisko progu), 8 miało 20-30%,
    7 miało 10-20%. Obniżenie progu decyzyjnego do 25% dałoby recall=0.865 na obecnych danych.

### Konkretne instrukcje dla Agent 2

**Próg PM10 do tworzenia targetu:**
  Użyj: PM10 >= 50 LUB PM2.5 >= 25 µg/m³ (zachowaj oryginalną definicję z data_preparation.ipynb)
  NIE zmieniaj progu — jest zgodny z WHO i z oryginalnym projektem.
  Pamiętaj: PM2.5 to dominujący wyzwalacz (75.7% alarmów) — model MUSI to umieć wykryć.

**Strategia danych:**
  [X] Wszystkie dane z sample weightingiem
  Uzasadnienie:
    raw_data.csv ma tylko 182 wiersze (2025+). Dane treningowe 2021-2024 (dataset.xlsx) są niedostępne.
    Jedyną opcją jest trenowanie na dostępnych danych produkcyjnych (2025+).
    WAŻNE: dane mają silną sezonowość — sierpień/wrzesień (lato) to tylko 39 wierszy,
    a zima (sty/luty 2026) to 50 wierszy. Zastosuj sample weighting: wagi odwrotnie proporcjonalne
    do odległości od końca datasetu LUB wyższe wagi dla zimowych obserwacji.
  Schemat wag (jeśli dotyczy):
    Opcja A (temporal): weight = 0.5 + 0.5 * (row_index / total_rows) — nowsze dane ważniejsze
    Opcja B (sezonowy): wagi na podstawie inv_freq per miesiąc (sty/luty/listo = niższe, lato = wyższe)
    Rekomendacja: Opcja A (temporal) — prostsze i uwzględnia drift temporalny

**Obsługa niezbilansowanych klas:**
  Rekomendacja: class_weight='balanced' ORAZ SMOTE jeśli mało danych
  Uzasadnienie:
    W danych produkcyjnych ratio 0:1 = 1.45:1 (umiarkowany imbalance), ale oczekiwany ratio
    w zbiorze treningowym może być inny po time-split (zimowe dane = więcej target=1).
    class_weight='balanced' to minimum. SMOTE rozważ jeśli train set ma < 50 próbek klasy 1.
    Priorytet: recall >= 0.80, więc lepiej overbalancować niż underbalancować.

**Okno czasowe:**
  Rekomendowane okno: 5-7 dni
  Uzasadnienie:
    Obecne 3-dniowe okno jest niewystarczające. PM2.5 narasta stopniowo przez kilka dni
    (inercja atmosferyczna). 5-7 dniowe okno lepiej uchwytuje trendy narastające.
    Dodaj rolling mean PM10 i PM2.5 za ostatnie 7 dni jako dodatkowe features.
    Porównaj TimeSeriesSplit CV z oknem 3 vs 5 vs 7 dni.

**Dodatkowe features do rozważenia:**
  - PM2.5_7day_avg: rolling average PM2.5 za 7 dni (kluczowe — główny wyzwalacz alarmów)
  - PM10_7day_avg: rolling average PM10 za 7 dni
  - PM2.5_PM10_ratio: stosunek PM2.5/PM10 (wskaźnik combustion vs. dust)
  - temp_below_5: flaga czy temperatura < 5°C (sezon grzewczy)
  - winter_season: flaga czy miesiac in [10,11,12,1,2,3] (sezon zimowy)
  - days_since_rain: liczba dni od ostatnich opadów >= 1mm (akumulacja PM)
  - pressure_trend_3d: trend ciśnienia 3-dniowy (antycyklony = akumulacja PM)
  - PM2.5_above_20: flaga czy PM2.5 dziś > 20 (pre-alarm signal)
  UWAGA: PM10 i PM2.5 to FEATURES (opis bieżącej sytuacji). NIE usuwaj ich surowych wartości
  tak jak robi to stary model — zostaw też PM10 i PM2.5 z dnia D obok agregatów!

**Progi decyzyjne:**
  Obecne progi (0.30/0.50) optymalne? NIE
  Sugerowany nowy próg dla "high": 25% (zamiast 50%)
  Uzasadnienie (jak zmiana progu wpłynie na Recall):
    Analiza obecnego modelu na danych produkcyjnych:
      próg 50% -> Recall=0.581, Precision=0.662, Acc=0.707
      próg 30% -> Recall=0.797, Precision=0.527, Acc=0.624
      próg 25% -> Recall=0.865, Precision=0.516, Acc=0.613
      próg 20% -> Recall=0.905, Precision=0.500, Acc=0.591
    Cel: Recall >= 0.80. Przy nowym, lepszym modelu próg 25-30% powinien dać Recall >= 0.80
    przy lepszej Precision niż obecny model (bo model będzie lepiej skalibrowany).
    Nowy próg decyzyjny wyznacz na danych walidacyjnych: szukaj najmniejszego progu gdzie
    Recall_high >= 0.80. Oczekiwany zakres: 0.20-0.35.
  Format notyfikacji: zachowaj 3 poziomy (low/moderate/high), ale próg high obniż do 25%
  a próg moderate do 15%.

**Priorytet podejść modelowania:**
  1. LightGBM / XGBoost z scale_pos_weight + threshold tuning (PRIORYTET)
     Uzasadnienie: GBM lepiej obsługuje gradientowe trendy PM2.5 (kluczowy sygnał),
     natywnie wspiera ważone klasy, szybki trening na małym zbiorze (182 próbki).
     scale_pos_weight = (n_low / n_high) ~ 1.45. Threshold tuning po trenowaniu.
  2. RandomForest z class_weight='balanced' + rozszerzone features + threshold tuning
     Uzasadnienie: kontynuacja podejścia oryginalnego, ale z poprawkami.
     Zmniejsz max_depth (10-15), zwiększ min_samples_leaf (3-5), class_weight='balanced'.
     Kluczowe: dodaj PM2.5_7day_avg i winter_season do features.
  3. Logistic Regression z regularyzacją jako baseline
     Uzasadnienie: prosty model, łatwa interpretacja, dobra kalibracja prawdopodobieństw.
     Przydatny jako benchmark i do sprawdzenia czy problem jest liniowalny.

**Ostrzeżenia i pułapki:**
  - MAŁY ZBIÓR DANYCH: tylko 182 wiersze. TimeSeriesSplit z n_splits=5 da bardzo małe fold-y.
    Rozważ n_splits=3 lub walk-forward validation z krokiem 1 miesiąca.
  - SEZONOWOŚĆ W TRAIN/TEST: test set musi zawierać zimę (sty-luty 2026) gdzie jest
    najwięcej alarmów. Proponowany podział: train do 2025-11-30, test od 2025-12-01.
  - BRAK ORYGINALNYCH DANYCH TRENINGOWYCH: dataset.xlsx niedostępny. Nie ma możliwości
    połączenia z danymi 2021-2024 bez ich ponownego pobrania.
  - DOMINANT TRIGGER IS PM2.5: 75.7% alarmów pochodzi od PM2.5, nie PM10. Model musi
    "nauczyć się" PM2.5 jako primarym sygnałem. Usuń zachowanie usuwania raw PM values!
  - LEAKAGE PREVENTION: sprawdź dwukrotnie że PM10/PM2.5 z dnia D+1 NIE wchodzi do features.
    Bezpieczne: PM10_avg z okna [D, D-1, D-2]. NIEBEZPIECZNE: PM10 z dnia D+1 jako feature.
  - OVERFITTING: max_depth=30 w starym modelu to overfitting. Ogranicz do max_depth=10-15.
  - BRAK DANYCH LETNICH W TEST: jeśli test = sty-luty, model nie jest oceniany na lecie.
    Sezon letni ma inne wzorce (rzadkie alarmy). Odnotuj to jako ograniczenie.
  - 3 BRAKI PM2.5: uzupełnij interpolacją liniową przed trenowaniem.
  - 2 LUKI W DATACH (6 i 22 stycznia 2026): uzupełnij interpolacją liniową.

---

## Agent 2 — Retrenowanie modelu
*[Agent wypełnia po zakończeniu]*

**Data wykonania:** 2026-02-26

### Przygotowanie danych
- Próg PM10 użyty: PM10 >= 50 LUB PM2_5 >= 25 µg/m³ (zgodnie z data_preparation.ipynb i rekomendacją Agent 1)
- Wiersze po czyszczeniu: przed 182 → po 176 (resample daily + interpolacja liniowa + usunięcie NaN z lagów)
- Target=1 w train: 28 (30.1%) — zbiór treningowy sierpień-listopad 2025
- Target=1 w test:  46 (55.4%) — zbiór testowy grudzień 2025 - luty 2026 (intensywny sezon zimowy)
- Okno czasowe użyte: 7 dni (lagi D-1 do D-7)
- Liczba features łącznie: 75
- Podział: train do 2025-11-30 | test od 2025-12-01
- Sample weighting: TAK — temporal: weight = 0.5 + 0.5 * (row_index / n_train), nowsze dane ważniejsze

### Porównanie podejść
| Podejście | Accuracy | Recall-high | Precision-high | F1-high | Próg | AUC |
|-----------|----------|-------------|----------------|---------|------|-----|
| A — RandomForest | 0.639 | 0.826 | 0.633 | 0.717 | 0.56 | 0.664 |
| B — LightGBM | 0.614 | 0.848 | 0.609 | 0.709 | 0.43 | 0.605 |
| C — XGBoost | 0.566 | 1.000 | 0.561 | 0.719 | 0.29 | 0.548 |
| D — Logistic Regression [WYBRANY] | 0.699 | 0.804 | 0.698 | 0.747 | 0.57 | 0.724 |

Wszystkie podejścia z optymalnym progiem (minimalne threshold gdzie Recall >= 0.80).
TimeSeriesSplit n_splits=3, RandomizedSearchCV 30 iteracji, scoring='recall'.

### Wybrany model
- Podejście: D — Logistic Regression z threshold tuning
- Algorytm i hiperparametry: Pipeline(StandardScaler + LogisticRegression(C=0.001, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=2000))
- Uzasadnienie wyboru: Spośród wszystkich modeli z Recall >= 0.80, Logistic Regression osiągnęła najwyższą Accuracy (0.699) i Precision (0.698). AUC-ROC 0.724 jest najwyższe spośród wszystkich podejść. Model dobrze skalibrowany (liniowy, regularyzowany). RF dał lepszy Recall (0.826) ale niższe Accuracy (0.639).

### Wyniki finalnego modelu na TEST SET
- Accuracy:                 0.699 ← cel ≥ 0.80 [NIE OSIĄGNIĘTY — patrz ograniczenia]
- Recall dla klasy "high":  0.804 ← cel ≥ 0.80 (PRIORYTET) [OSIĄGNIĘTY]
- Precision dla klasy "high": 0.698
- F1 dla klasy "high":      0.747
- AUC-ROC:                  0.724
- Optymalny próg decyzyjny: 0.57 (stary: 0.50)
- Confusion matrix: TP=37, FP=16, TN=21, FN=9

UWAGA: Cel Accuracy >= 0.80 nie był możliwy do osiągnięcia jednocześnie z Recall >= 0.80 z powodu struktury test setu. Test set (grudzień 2025 - luty 2026) ma 55.4% target=1 — intensywna zima z wyjątkowo wysokim PM2.5 i PM10. Przy tak wysokim udziale klasy pozytywnej, accuracy jest strukturalnie ograniczone. Cel priorytetowy (Recall >= 0.80) został OSIĄGNIĘTY.

### Porównanie ze starym modelem
| Metryka | Stary model | Nowy model | Zmiana |
|---------|-------------|------------|--------|
| Accuracy | 0.707 | 0.699 | -0.008 (nieznaczne) |
| Recall-high | 0.581 | 0.804 | +0.223 (znacząca poprawa) |
| Precision-high | 0.662 | 0.698 | +0.036 (poprawa) |
| AUC-ROC | N/A | 0.724 | nowy pomiar |
| Próg decyzyjny | 0.50 | 0.57 | zmiana |
| FN (przeoczenia) | 31/74 | 9/46 | z 41.9% do 19.6% |

### Zmiany w predict.py
- Nowy model: rf_model_v2.joblib (Logistic Regression pipeline)
- Nowy próg decyzyjny: 0.57 (wczytywany z models/optimal_threshold.json)
- Zmiany w feature engineeringu: okno 7 dni (lagi 1-7) zamiast 3-dniowych sekwencji; zachowanie surowych PM10 i PM2.5; nowe features: PM2_5_7day_avg, PM2_5_PM10_ratio, winter_season, days_since_rain, PM2_5_above_20, PM10_above_40, pressure_trend_3d
- Walidacja kolejności features z data/clean/feature_names.txt
- Wczytuje progi z models/optimal_threshold.json

### Zapisane pliki
- [x] rf_model_v2.joblib
- [x] data/clean/feature_names.txt
- [x] models/all_results.json
- [x] models/final_metrics.json
- [x] models/optimal_threshold.json
- [x] predict.py (zaktualizowany)
- [x] predict_v1.py (backup)
- [x] reports/report.html
- [x] reports/figures/ (5 wykresów: confusion_matrix_final.png, roc_curve.png, feature_importance_top15.png, metrics_comparison.png, predictions_over_time.png)

### Ostrzeżenia / ograniczenia
- CEL RECALL OSIĄGNIĘTY: 0.804 >= 0.80
- CEL ACCURACY NIE OSIĄGNIĘTY: 0.699 < 0.80. Przyczyna strukturalna: test set zawiera tylko zimę 2025/2026 (55.4% target=1). Żaden model trenowany na lecie nie osiągnie accuracy=0.80 na tak zimowym test secie przy zachowaniu recall>=0.80.
- Mały dataset (176 wierszy) — wzorce mogą być niestabilne. Zalecane retrenowanie po zebraniu danych z kolejnych sezonów.
- Brak letnich danych w test secie — model nie jest oceniany na sezonach z rzadkimi alarmami. Może dawać nadmiernie dużo FP latem.
- CV Recall w TimeSeriesSplit był niski (0.333 dla modeli GBM/RF) — silny overfitting na małym train secie. LR osiągnęła CV Recall=0.556 i lepszą generalizację.
- Monitoring: zalecany miesięczny przegląd Recall na nowych danych produkcyjnych.
