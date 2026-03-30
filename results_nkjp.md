# SSD Method Comparison Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding | nkjp+wiki CBOW 300d (L2 + ABTT m=1) |
| Context window | ±3 tokens |
| SIF a | 0.001 |
| **SSDLite PLS** | pls |
| **SSDLite PCA+OLS** | pcaols |
| **Official (PCA)** | official |

## Per-Dataset Results

### imigrant

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 17 | 17 |
| R² | 0.1548 | 0.0489 | 0.0489 |
| adj R² | 0.1529 | 0.0117 | 0.0117 |
| p-value | 0.05550 | 0.17854 | 0.17854 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 69.2% | 69.2% | 69.2% |
| Time (s) | 0.0 | 24.9 | 43.7 |

### klimat

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 30 | 30 |
| R² | 0.1407 | 0.1146 | 0.1146 |
| adj R² | 0.1392 | 0.0649 | 0.0649 |
| p-value | 0.00002 | 0.00013 | 0.00013 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 86.3% | 86.3% | 86.3% |
| Time (s) | 0.1 | 29.1 | 50.2 |

### naukowcy

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 21 | 21 |
| R² | 0.0999 | 0.0801 | 0.0801 |
| adj R² | 0.0984 | 0.0465 | 0.0465 |
| p-value | 0.00003 | 0.00054 | 0.00054 |
| N (observations) | 648 | 648 | 648 |
| Coverage | 92.1% | 92.1% | 92.1% |
| Time (s) | 0.1 | 29.8 | 51.4 |

### polityka

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 7 | 7 |
| R² | 0.1421 | 0.0938 | 0.0938 |
| adj R² | 0.1406 | 0.0827 | 0.0827 |
| p-value | 0.00000 | 0.00000 | 0.00000 |
| N (observations) | 648 | 648 | 648 |
| Coverage | 89.5% | 89.5% | 89.5% |
| Time (s) | 0.1 | 30.1 | 50.9 |

### szczepienie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 18 | 18 |
| R² | 0.1873 | 0.0824 | 0.0824 |
| adj R² | 0.1859 | 0.0538 | 0.0538 |
| p-value | 0.00003 | 0.00007 | 0.00007 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 91.1% | 91.1% | 91.1% |
| Time (s) | 0.1 | 29.9 | 47.3 |

### zaufanie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 30 | 30 |
| R² | 0.1182 | 0.0886 | 0.0886 |
| adj R² | 0.1167 | 0.0401 | 0.0401 |
| p-value | 0.00013 | 0.00512 | 0.00512 |
| N (observations) | 636 | 636 | 636 |
| Coverage | 93.5% | 93.5% | 93.5% |
| Time (s) | 0.0 | 31.7 | 52.7 |

### zdrowie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 21 | 21 |
| R² | 0.1415 | 0.0678 | 0.0678 |
| adj R² | 0.1399 | 0.0301 | 0.0301 |
| p-value | 0.01080 | 0.01634 | 0.01634 |
| N (observations) | 636 | 636 | 636 |
| Coverage | 85.1% | 85.1% | 85.1% |
| Time (s) | 0.1 | 27.2 | 46.6 |

## Summary

| Dataset | K_PLS | R²_PLS | adj_PLS | K_PCA+OLS | R²_PCA+OLS | adj_PCA+OLS | K_(PCA) | R²_(PCA) | adj_(PCA) |
|---|---|---|---|---|---|---|---|---|---|
| imigrant | 1 | 0.1548 | 0.1529 | 17 | 0.0489 | 0.0117 | 17 | 0.0489 | 0.0117 |
| klimat | 1 | 0.1407 | 0.1392 | 30 | 0.1146 | 0.0649 | 30 | 0.1146 | 0.0649 |
| naukowcy | 1 | 0.0999 | 0.0984 | 21 | 0.0801 | 0.0465 | 21 | 0.0801 | 0.0465 |
| polityka | 1 | 0.1421 | 0.1406 | 7 | 0.0938 | 0.0827 | 7 | 0.0938 | 0.0827 |
| szczepienie | 1 | 0.1873 | 0.1859 | 18 | 0.0824 | 0.0538 | 18 | 0.0824 | 0.0538 |
| zaufanie | 1 | 0.1182 | 0.1167 | 30 | 0.0886 | 0.0401 | 30 | 0.0886 | 0.0401 |
| zdrowie | 1 | 0.1415 | 0.1399 | 21 | 0.0678 | 0.0301 | 21 | 0.0678 | 0.0301 |

## Top Words Comparison

### imigrant

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | motywacja | motywacja | motywacja |
| 2 | doświadczenie | doświadczenie | doświadczenie |
| 3 | przeżycie | nastawienie | nastawienie |
| 4 | odczucie | cecha | cecha |
| 5 | uczucie | warunki | warunki |
| 6 | cecha | podejście | podejście |
| 7 | nastawienie | odczucie | odczucie |
| 8 | aktywność | zaleta | zaleta |
| 9 | doznanie | zdolność | zdolność |
| 10 | przekonanie | kwalifikacja | kwalifikacja |
| 11 | emocja | osiągnięcie | osiągnięcie |
| 12 | życzliwość | wykształcenie | wykształcenie |
| 13 | świadomość | upodobanie | upodobanie |
| 14 | wysiłek | przekonanie | przekonanie |
| 15 | wrażliwość | kondycja | kondycja |
| 16 | aspekt | kwalifikacje | kwalifikacje |
| 17 | satysfakcja | preferencja | preferencja |
| 18 | problem | wiedza | wiedza |
| 19 | sympatia | skłonność | skłonność |
| 20 | troska | umiejętność | umiejętność |

All-way overlap: 18%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | spalić | odsunąć | odsunąć |
| 2 | zaatakować | odciąć | odciąć |
| 3 | wstrzymać | odwrócić | odwrócić |
| 4 | rozbić | oderwać | oderwać |
| 5 | przewrócić | spalić | spalić |
| 6 | uszkodzić | zmienić | zmienić |
| 7 | zniszczyć | powstrzymać | powstrzymać |
| 8 | odciąć | zamknąć | zamknąć |
| 9 | podpalić | straszliwy | straszliwy |
| 10 | urwać | zerwać | zerwać |
| 11 | powstrzymać | uwolnić | uwolnić |
| 12 | uszkodzony | rozbić | rozbić |
| 13 | wycofać | wstrzymać | wstrzymać |
| 14 | rzucić | zaostrzyć | zaostrzyć |
| 15 | podnieść | krwawy | krwawy |
| 16 | wystrzelić | cofnąć | cofnąć |
| 17 | zatrzymać | zniszczyć | zniszczyć |
| 18 | zawalić | wycofać | wycofać |
| 19 | rozpocząć | odbić | odbić |
| 20 | cofnąć | odłączyć | odłączyć |

All-way overlap: 25%

### klimat

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | upał | komfort | komfort |
| 2 | mróz | poczucie | poczucie |
| 3 | zima | troska | troska |
| 4 | zmęczenie | stres | stres |
| 5 | deszcz | satysfakcja | satysfakcja |
| 6 | śnieg | zmęczenie | zmęczenie |
| 7 | stres | samopoczucie | samopoczucie |
| 8 | opad | rywal | rywal |
| 9 | powódź | wygoda | wygoda |
| 10 | awaria | pogoda | pogoda |
| 11 | zator | chęć | chęć |
| 12 | niebezpieczeństwo | defensywa | defensywa |
| 13 | pogoda | ostrzeżenie | ostrzeżenie |
| 14 | woda | niebezpieczeństwo | niebezpieczeństwo |
| 15 | korek | kierowca | kierowca |
| 16 | drogowiec | ulga | ulga |
| 17 | powietrze | obawa | obawa |
| 18 | kłopot | warunki | warunki |
| 19 | hałas | motywacja | motywacja |
| 20 | pośpiech | smutek | smutek |

All-way overlap: 11%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | historyczny | ewolucja | ewolucja |
| 2 | przedstawiać | historyczny | historyczny |
| 3 | filozofia | rekonstrukcja | rekonstrukcja |
| 4 | znawca | istnienie | istnienie |
| 5 | abstrakcyjny | synteza | synteza |
| 6 | twórca | przeciez | przeciez |
| 7 | negować | badać | badać |
| 8 | biblijny | opisywać | opisywać |
| 9 | zasadniczo | przedstawiać | przedstawiać |
| 10 | wybitny | negować | negować |
| 11 | fundamentalny | słynny | słynny |
| 12 | odrębny | dzieje | dzieje |
| 13 | słynny | geologiczny | geologiczny |
| 14 | opisywać | omawiać | omawiać |
| 15 | uniwersalny | archeologiczny | archeologiczny |
| 16 | wedle | abstrakcyjny | abstrakcyjny |
| 17 | historyk | zasadniczo | zasadniczo |
| 18 | ujmować | przerabiać | przerabiać |
| 19 | kwestionować | odrębny | odrębny |
| 20 | kreować | kwestionować | kwestionować |

All-way overlap: 29%

### naukowcy

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | zaangażowanie | dokonanie | dokonanie |
| 2 | różnorodność | umiejętność | umiejętność |
| 3 | twórczy | osiągnięcie | osiągnięcie |
| 4 | materialny | zaangażowanie | zaangażowanie |
| 5 | duchowy | rzetelny | rzetelny |
| 6 | intelektualny | twórczy | twórczy |
| 7 | dążenie | wrażliwość | wrażliwość |
| 8 | bogactwo | gratulować | gratulować |
| 9 | rozwój | życzliwość | życzliwość |
| 10 | poznawczy | docenić | docenić |
| 11 | społeczny | zdolność | zdolność |
| 12 | naukowy | intelektualny | intelektualny |
| 13 | życzliwość | talent | talent |
| 14 | rzetelny | uczciwość | uczciwość |
| 15 | kształtować | dążenie | dążenie |
| 16 | praktyczny | osobowość | osobowość |
| 17 | wzajemny | uznanie | uznanie |
| 18 | doskonalić | samodzielność | samodzielność |
| 19 | różnorodny | wierność | wierność |
| 20 | aktywność | wiarygodność | wiarygodność |

All-way overlap: 18%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | koleś | podobno | podobno |
| 2 | kumpel | akurat | akurat |
| 3 | podobno | znowu | znowu |
| 4 | juz | gdzieś | gdzieś |
| 5 | znowu | ostatnio | ostatnio |
| 6 | zaraz | bodajże | bodajże |
| 7 | akurat | barak | barak |
| 8 | kurwa | już | już |
| 9 | xD | prawdopodobnie | prawdopodobnie |
| 10 | kiedys | międzyczas | międzyczas |
| 11 | chłopak | nieczynny | nieczynny |
| 12 | budka | zaraz | zaraz |
| 13 | facet | znów | znów |
| 14 | kulka | nagle | nagle |
| 15 | przypadkowo | kiedyś | kiedyś |
| 16 | pijany | koleś | koleś |
| 17 | kiedyś | teraz | teraz |
| 18 | bodajże | tydzień | tydzień |
| 19 | cos | budka | budka |
| 20 | znów | noc | noc |

All-way overlap: 29%

### polityka

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | samorządowy | uczestnictwo | uczestnictwo |
| 2 | uczestnictwo | ramy | ramy |
| 3 | powiatowy | trakt | trakt |
| 4 | ponowny | udział | udział |
| 5 | wojewódzki | powiatowy | powiatowy |
| 6 | senat | uczestnik | uczestnik |
| 7 | ramy | przeddzień | przeddzień |
| 8 | samorząd | ponowny | ponowny |
| 9 | sejm | sekretariat | sekretariat |
| 10 | regionalny | wtorek | wtorek |
| 11 | projekt | coroczny | coroczny |
| 12 | parlamentarny | czwartek | czwartek |
| 13 | krajowy | podczas | podczas |
| 14 | gotowość | wojewódzki | wojewódzki |
| 15 | komisja | piątek | piątek |
| 16 | parlament | środa | środa |
| 17 | zgromadzenie | czerwiec | czerwiec |
| 18 | małopolski | ponownie | ponownie |
| 19 | członkostwo | samorządowy | samorządowy |
| 20 | edukacja | następnie | następnie |

All-way overlap: 18%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | znać | śmieszny | śmieszny |
| 2 | śmieszny | głupi | głupi |
| 3 | mylić | niby | niby |
| 4 | widzieć | chyba | chyba |
| 5 | czepiać | wiedzieć | wiedzieć |
| 6 | wiedzieć | rozumieć | rozumieć |
| 7 | przesadzać | gadać | gadać |
| 8 | niby | widzieć | widzieć |
| 9 | żartować | lubić | lubić |
| 10 | zdziwić | boleć | boleć |
| 11 | pomylić | straszny | straszny |
| 12 | męczyć | znać | znać |
| 13 | straszny | naiwny | naiwny |
| 14 | gadać | czepiać | czepiać |
| 15 | wogóle | szkodzić | szkodzić |
| 16 | kojarzyć | brzydki | brzydki |
| 17 | lubić | okropny | okropny |
| 18 | denerwować | żałosny | żałosny |
| 19 | śmiać | znaczyć | znaczyć |
| 20 | nudzić | oczywisty | oczywisty |

All-way overlap: 29%

### szczepienie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | przejaw | sprzyjać | sprzyjać |
| 2 | cywilizacyjny | zwolennik | zwolennik |
| 3 | niewątpliwie | zawdzięczać | zawdzięczać |
| 4 | dzięki | dzięki | dzięki |
| 5 | idea | wśród | wśród |
| 6 | społeczeństwo | umożliwić | umożliwić |
| 7 | rozwój | wspierać | wspierać |
| 8 | duchowy | zaowocować | zaowocować |
| 9 | sprzyjać | zasługa | zasługa |
| 10 | sfera | umożliwiać | umożliwiać |
| 11 | priorytet | pragnąć | pragnąć |
| 12 | zasługa | idea | idea |
| 13 | społeczność | inicjator | inicjator |
| 14 | dziedzina | pomagać | pomagać |
| 15 | problematyka | dla | dla |
| 16 | aspekt | region | region |
| 17 | zwolennik | niewątpliwie | niewątpliwie |
| 18 | strategia | sojusznik | sojusznik |
| 19 | poprzez | wspomagać | wspomagać |
| 20 | element | przejaw | przejaw |

All-way overlap: 21%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | wpuścić | cofnąć | cofnąć |
| 2 | przesłuchać | zadzwonić | zadzwonić |
| 3 | pogadać | wstawić | wstawić |
| 4 | zadzwonić | przesłuchać | przesłuchać |
| 5 | polecieć | zajrzeć | zajrzeć |
| 6 | zastrzelić | odwołać | odwołać |
| 7 | przerobić | zbadać | zbadać |
| 8 | dzwonić | sprawdzić | sprawdzić |
| 9 | wyjechać | wpis | wpis |
| 10 | sprawdzić | odesłać | odesłać |
| 11 | wylądować | dzwonić | dzwonić |
| 12 | spytać | przesunąć | przesunąć |
| 13 | przepisać | pobrać | pobrać |
| 14 | jakis | podać | podać |
| 15 | wrócić | przejechać | przejechać |
| 16 | pojechać | skierowanie | skierowanie |
| 17 | przesłuchanie | wrzucić | wrzucić |
| 18 | zameldować | przedłużyć | przedłużyć |
| 19 | zawieźć | przejrzeć | przejrzeć |
| 20 | podać | uzupełnić | uzupełnić |

All-way overlap: 14%

### zaufanie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | atrakcyjny | popularny | popularny |
| 2 | popularny | atrakcyjny | atrakcyjny |
| 3 | zróżnicować | ciekawie | ciekawie |
| 4 | liczny | zróżnicować | zróżnicować |
| 5 | różnorodny | bogaty | bogaty |
| 6 | dobry | dobry | dobry |
| 7 | bogaty | nowoczesny | nowoczesny |
| 8 | przyjazny | fajny | fajny |
| 9 | zaawansowany | utalentowany | utalentowany |
| 10 | tradycyjny | interesujący | interesujący |
| 11 | niewielki | modny | modny |
| 12 | specyficzny | aktywny | aktywny |
| 13 | różny | zaawansowany | zaawansowany |
| 14 | nowoczesny | prestiżowy | prestiżowy |
| 15 | sympatyczny | sympatyczny | sympatyczny |
| 16 | korzystny | odległy | odległy |
| 17 | aktywny | udany | udany |
| 18 | duży | pracowity | pracowity |
| 19 | skromny | imponujący | imponujący |
| 20 | szeroki | wspaniały | wspaniały |

All-way overlap: 29%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | odeprzeć | nakazać | nakazać |
| 2 | krzyczeć | krzyczeć | krzyczeć |
| 3 | widzieć | kazać | kazać |
| 4 | krzyknąć | odeprzeć | odeprzeć |
| 5 | oświadczyć | zarzucić | zarzucić |
| 6 | uwierzyć | nakazywać | nakazywać |
| 7 | oznajmić | usprawiedliwiać | usprawiedliwiać |
| 8 | poczuć | odmawiać | odmawiać |
| 9 | wytłumaczyć | wytłumaczyć | wytłumaczyć |
| 10 | zawołać | oświadczyć | oświadczyć |
| 11 | zauważyć | zdradzić | zdradzić |
| 12 | powiadać | straszyć | straszyć |
| 13 | zdradzić | krzyknąć | krzyknąć |
| 14 | udowodnić | odmówić | odmówić |
| 15 | wierzyć | lekceważyć | lekceważyć |
| 16 | zarzucić | wzbudzić | wzbudzić |
| 17 | powiedzieć | zdradzać | zdradzać |
| 18 | stwierdzić | natychmiast | natychmiast |
| 19 | przypuszczać | zabić | zabić |
| 20 | zaprzeczyć | grozić | grozić |

All-way overlap: 21%

### zdrowie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | stwórca | bóg | bóg |
| 2 | harmonia | miłować | miłować |
| 3 | koniecznie | stwórca | stwórca |
| 4 | bóg | piękno | piękno |
| 5 | niepowtarzalny | harmonia | harmonia |
| 6 | zrobic | chwała | chwała |
| 7 | spróbować | dusza | dusza |
| 8 | hehe | niebo | niebo |
| 9 | spokój | zbawienie | zbawienie |
| 10 | wspaniały | miłość | miłość |
| 11 | kochany | dobroć | dobroć |
| 12 | mądrość | rozum | rozum |
| 13 | zbawienie | niepowtarzalny | niepowtarzalny |
| 14 | nns | smaczny | smaczny |
| 15 | śliczny | jasność | jasność |
| 16 | miłość | zbawić | zbawić |
| 17 | piękno | wspaniały | wspaniały |
| 18 | zycia | mądrość | mądrość |
| 19 | bedziesz | zdrowy | zdrowy |
| 20 | talent | mądry | mądry |

All-way overlap: 29%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | obecnie | przeważnie | przeważnie |
| 2 | szczególnie | nierzadko | nierzadko |
| 3 | przeważnie | wprawdzie | wprawdzie |
| 4 | kilkakrotnie | kilkakrotnie | kilkakrotnie |
| 5 | niejednokrotnie | niejednokrotnie | niejednokrotnie |
| 6 | wielokrotnie | dość | dość |
| 7 | nierzadko | dosyć | dosyć |
| 8 | nieoficjalnie | ostatnio | ostatnio |
| 9 | zazwyczaj | sporadycznie | sporadycznie |
| 10 | sporadycznie | pomijać | pomijać |
| 11 | często | niekiedy | niekiedy |
| 12 | powszechnie | szczególnie | szczególnie |
| 13 | tymczasem | głównie | głównie |
| 14 | nadal | niektóry | niektóry |
| 15 | wyjątkowo | wielokrotnie | wielokrotnie |
| 16 | ostatnio | niezbyt | niezbyt |
| 17 | niekiedy | początkowo | początkowo |
| 18 | wciąż | obecnie | obecnie |
| 19 | pomijać | tymczasem | tymczasem |
| 20 | zwłaszcza | wskutek | wskutek |

All-way overlap: 43%

## Clusters Comparison

### imigrant

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 35 | 0.556 | 0.555 | doświadczenie, cecha, predyspozycja, aspekt, sposób |
| pos | 2 | 65 | 0.584 | 0.541 | uczucie, wrażliwość, odczucie, krytycyzm, seksualność |
| neg | 1 | 56 | 0.590 | -0.423 | zniszczyć, ostrzelać, zaatakować, rozbić, spalić |
| neg | 2 | 44 | 0.646 | -0.387 | odbić, przewrócić, urwać, odciąć, oderwać |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 59 | 0.555 | 0.652 | nastawienie, odczucie, światopogląd, pogląd, cecha |
| pos | 2 | 41 | 0.584 | 0.644 | predyspozycja, zdolność, umiejętność, motywacja, potencjał |
| neg | 1 | 49 | 0.482 | -0.582 | zniszczyć, spacyfikować, straszliwy, zdławić, zdruzgotać |
| neg | 2 | 51 | 0.657 | -0.452 | rozlecieć, rozsypać, odciąć, urwać, oderwać |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 67 | 0.552 | 0.668 | zdolność, motywacja, predyspozycja, potencjał, umiejętność |
| pos | 2 | 33 | 0.600 | 0.610 | odczucie, nastawienie, przekonanie, pogląd, mniemanie |
| neg | 1 | 29 | 0.521 | -0.537 | haniebny, skandaliczny, straszliwy, bezprecedensowy, dramatyczny |
| neg | 2 | 71 | 0.616 | -0.476 | odciąć, rozsypać, rozerwać, rozbić, urwać |

### klimat

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 44 | 0.610 | 0.513 | upał, ulewa, śnieżyca, deszcz, przymrozek |
| pos | 2 | 18 | 0.544 | 0.505 | utrudnienie, nawierzchnia, odwodnienie, drogowiec, korek |
| pos | 3 | 38 | 0.596 | 0.469 | zmęczenie, stres, przemęczenie, omdlenie, duszność |
| neg | 1 | 21 | 0.592 | -0.510 | ideologia, idealistyczny, postępowy, ideowy, lewacki |
| neg | 2 | 29 | 0.617 | -0.509 | abstrakcyjny, postmodernistyczny, ponadczasowy, antropologiczny, współczesny |
| neg | 3 | 27 | 0.640 | -0.483 | opisywać, przedstawiać, definiować, negować, dyskredytować |
| neg | 4 | 9 | 0.648 | -0.470 | rozne, rózne, istnieja, róznych, zadna |
| neg | 5 | 14 | 0.657 | -0.468 | twórca, wybitny, teoretyk, historyk, współtwórca |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 38 | 0.442 | 0.528 | maluch, pasażer, posiłek, jedzenie, posiłki |
| pos | 2 | 62 | 0.526 | 0.472 | zmęczenie, stres, zdenerwowanie, przygnębienie, rozdrażnienie |
| neg | 1 | 49 | 0.511 | -0.571 | ewolucja, historyczny, socjologiczny, antropologiczny, geneza |
| neg | 2 | 51 | 0.530 | -0.535 | opisywać, analizować, przedstawiać, przytaczać, opisać |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 52 | 0.418 | 0.567 | jedzenie, maluch, posiłek, pasażer, wczasowicz |
| pos | 2 | 48 | 0.572 | 0.434 | zmęczenie, przygnębienie, zdenerwowanie, rozdrażnienie, irytacja |
| neg | 1 | 21 | 0.579 | -0.509 | ewolucja, badacz, geneza, ewolucyjny, kosmologia |
| neg | 2 | 29 | 0.579 | -0.491 | opisywać, rekonstruować, analizować, przedstawiać, opisać |
| neg | 3 | 19 | 0.606 | -0.475 | negować, wydumać, bzdurny, nieweryfikowalny, kwestionować |
| neg | 4 | 18 | 0.603 | -0.474 | socjologiczny, naukowy, historyczny, empiryczny, antropologiczny |
| neg | 5 | 13 | 0.685 | -0.416 | rozne, istnieja, roznych, odnosnie, zadna |

### naukowcy

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 49 | 0.576 | 0.688 | poznawczy, intelektualny, kształtować, upowszechniać, twórczy |
| pos | 2 | 51 | 0.594 | 0.670 | otwartość, kreatywność, profesjonalizm, zaangażowanie, wrażliwość |
| neg | 1 | 14 | 0.549 | -0.509 | badziewie, trampka, syf, parasolka, kulka |
| neg | 2 | 13 | 0.607 | -0.475 | knajpa, dyskoteka, piwko, melina, kibel |
| neg | 3 | 19 | 0.619 | -0.463 | znowu, głupio, kiedyś, fajnie, pewnie |
| neg | 4 | 25 | 0.656 | -0.459 | heh, koles, kurde, xD, kurcz |
| neg | 5 | 29 | 0.684 | -0.432 | gówniarz, łobuz, palant, koleś, skurwiel |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 20 | 0.575 | 0.679 | nieprzeciętny, ponadprzeciętny, wszechstronny, docenić, niepospolity |
| pos | 2 | 55 | 0.652 | 0.630 | pracowitość, profesjonalizm, wszechstronność, kreatywność, pomysłowość |
| pos | 3 | 14 | 0.649 | 0.562 | samoświadomość, zdolność, niezależność, samodzielność, dążenie |
| pos | 4 | 11 | 0.723 | 0.532 | obiektywność, rzetelność, prawdziwość, trafność, wiarygodność |
| neg | 1 | 62 | 0.482 | -0.561 | podobno, kiedyś, znowu, teraz, ponoć |
| neg | 2 | 38 | 0.509 | -0.516 | barak, rudera, komin, korytarz, budynek |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 29 | 0.539 | 0.718 | wszechstronny, nieprzeciętny, ponadprzeciętny, intelektualny, twórczy |
| pos | 2 | 71 | 0.621 | 0.646 | profesjonalizm, pracowitość, kreatywność, wszechstronność, fachowość |
| neg | 1 | 17 | 0.514 | -0.501 | pusto, opustoszały, ciemno, noc, śmierdzący |
| neg | 2 | 24 | 0.545 | -0.483 | syf, dziadostwo, kicha, badziewie, paskudztwo |
| neg | 3 | 28 | 0.591 | -0.480 | znowu, kiedyś, znów, teraz, podobno |
| neg | 4 | 24 | 0.588 | -0.448 | barak, rudera, budynek, kurnik, komin |
| neg | 5 | 7 | 0.728 | -0.348 | sporadycznie, nagminnie, przeważnie, notorycznie, masowo |

### polityka

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 39 | 0.510 | 0.591 | sejm, senat, komisja, parlament, uchwała |
| pos | 2 | 30 | 0.547 | 0.530 | instytucja, samorząd, przedsiębiorczość, organizacja, zakres |
| pos | 3 | 31 | 0.573 | 0.515 | małopolski, koniński, oświęcimski, powiatowy, gminny |
| neg | 1 | 35 | 0.653 | -0.688 | wiedzieć, kumać, widzieć, rozumieć, zrozumieć |
| neg | 2 | 15 | 0.707 | -0.620 | okropny, śmieszny, głupi, straszny, żałosny |
| neg | 3 | 50 | 0.738 | -0.605 | denerwować, irytować, podobać, podniecać, złościć |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 20 | 0.489 | 0.678 | trakt, ramy, przeddzień, uczestnictwo, ponowny |
| pos | 2 | 14 | 0.520 | 0.557 | biuro, komitet, starostwo, regionalny, sekretariat |
| pos | 3 | 23 | 0.631 | 0.468 | doroczny, dwudniowy, czwartkowy, trzydniowy, piątkowy |
| pos | 4 | 27 | 0.626 | 0.463 | tczewski, trzebiński, malborski, rumski, bocheński |
| pos | 5 | 16 | 0.839 | 0.367 | listopad, październik, sierpień, czerwiec, kwiecień |
| neg | 1 | 30 | 0.686 | -0.695 | wiedzieć, myśle, mówie, myśleć, kumać |
| neg | 2 | 36 | 0.677 | -0.691 | głupi, wredny, żałosny, śmieszny, paskudny |
| neg | 3 | 34 | 0.696 | -0.668 | podniecać, irytować, wkurzać, drażnić, czepiać |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 20 | 0.481 | 0.650 | ponowny, ramy, niezwłoczny, magistrat, sekretariat |
| pos | 2 | 32 | 0.607 | 0.515 | czwartkowy, piątkowy, poniedziałkowy, wtorkowy, dwudniowy |
| pos | 3 | 15 | 0.587 | 0.501 | powiatowy, wojewódzki, regionalny, małopolski, gminny |
| pos | 4 | 23 | 0.613 | 0.464 | tczewski, rumski, malborski, trzebiński, bocheński |
| pos | 5 | 10 | 0.990 | 0.303 | październik, kwiecień, marzec, listopad, lipiec |
| neg | 1 | 65 | 0.651 | -0.721 | wkurzać, podniecać, kumać, czepiać, irytować |
| neg | 2 | 35 | 0.680 | -0.691 | głupi, żałosny, wredny, śmieszny, infantylny |

### szczepienie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 33 | 0.522 | 0.569 | przejaw, idea, społeczeństwo, dynamizm, symbol |
| pos | 2 | 41 | 0.566 | 0.507 | rozwój, innowacyjność, upowszechniać, innowacja, przedsiębiorczość |
| pos | 3 | 26 | 0.598 | 0.481 | warunkować, nieodzowny, sprzyjać, nieunikniony, ułatwiać |
| neg | 1 | 43 | 0.588 | -0.449 | pogadać, pogadac, pisac, sprawdzic, chodzic |
| neg | 2 | 57 | 0.626 | -0.425 | zawieźć, wpuścić, podrzucić, polecieć, odesłać |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 32 | 0.545 | 0.578 | propagator, orędownik, piewca, entuzjasta, prekursor |
| pos | 2 | 21 | 0.533 | 0.562 | niewątpliwie, prestiż, dynamizm, bogactwo, dobrobyt |
| pos | 3 | 31 | 0.605 | 0.530 | sprzyjać, gwarantować, wspomagać, ułatwiać, umożliwić |
| pos | 4 | 16 | 0.561 | 0.524 | społeczność, społeczeństwo, środowisko, idea, diaspora |
| neg | 1 | 37 | 0.580 | -0.555 | wsiąść, zadzwonić, dojechać, podjechać, jechac |
| neg | 2 | 63 | 0.601 | -0.543 | wstawić, wrzucić, wyczyścić, podmienić, skorygować |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 45 | 0.509 | 0.606 | propagator, współtwórca, orędownik, entuzjasta, prekursor |
| pos | 2 | 26 | 0.544 | 0.552 | społeczeństwo, społeczność, dobrobyt, bogactwo, idea |
| pos | 3 | 29 | 0.611 | 0.526 | ułatwiać, umożliwić, umożliwiać, sprzyjać, gwarantować |
| neg | 1 | 55 | 0.592 | -0.550 | wyczyścić, usunąć, skorygować, załadować, podmienić |
| neg | 2 | 45 | 0.593 | -0.546 | zajrzeć, zaglądnąć, przyczepić, doczepić, zadzwonić |

### zaufanie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 45 | 0.493 | 0.498 | różnorodny, zróżnicować, specyficzny, nowoczesny, różny |
| pos | 2 | 55 | 0.578 | 0.441 | atrakcyjny, popularny, skromny, rzadki, obiecujący |
| neg | 1 | 28 | 0.693 | -0.550 | krzyknąć, zawołać, wrzasnąć, szepnąć, wykrzyknąć |
| neg | 2 | 72 | 0.713 | -0.537 | stwierdzić, udowadniać, powiedzieć, dowieść, sugerować |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 52 | 0.592 | 0.501 | ekscytujący, sympatyczny, obiecujący, fascynujący, ciekawy |
| pos | 2 | 48 | 0.603 | 0.481 | zróżnicować, atrakcyjny, efektywny, różnorodny, stabilny |
| neg | 1 | 42 | 0.616 | -0.529 | krzyknąć, krzyczeć, wrzasnąć, zawołać, poganiać |
| neg | 2 | 58 | 0.632 | -0.523 | zarzucić, stwierdzić, tłumaczyć, oświadczyć, potwierdzić |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 43 | 0.627 | 0.485 | obiecujący, ekscytujący, sympatyczny, atrakcyjny, świetny |
| pos | 2 | 40 | 0.605 | 0.474 | zróżnicować, różnorodny, efektywny, bogaty, stabilny |
| pos | 3 | 17 | 0.611 | 0.463 | stylowy, urokliwy, malowniczy, nowoczesny, eklektyczny |
| neg | 1 | 32 | 0.630 | -0.511 | poganiać, przepuścić, popchnąć, popędzać, ogłuszyć |
| neg | 2 | 17 | 0.650 | -0.509 | usprawiedliwiać, spowodować, wymusić, uzasadniać, powodować |
| neg | 3 | 15 | 0.733 | -0.456 | krzyknąć, zawołać, wrzasnąć, warknąć, krzyczeć |
| neg | 4 | 33 | 0.729 | -0.449 | oświadczyć, stwierdzić, oświadczać, tłumaczyć, zarzucić |
| neg | 5 | 3 | 0.834 | -0.438 | zważać, baczyć, omal |

### zdrowie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 31 | 0.522 | 0.500 | miłość, piękno, miłosci, bóg, miłośc |
| pos | 2 | 69 | 0.570 | 0.458 | hehe, heheh, haha, hihi, nio |
| neg | 1 | 38 | 0.455 | -0.550 | nasilić, przeciążyć, nieskuteczny, nasilać, załamać |
| neg | 2 | 24 | 0.552 | -0.476 | relatywnie, wyjątkowo, nadmiernie, stosunkowo, szczególnie |
| neg | 3 | 38 | 0.605 | -0.451 | często, niejednokrotnie, przeważnie, zazwyczaj, wielokrotnie |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 38 | 0.572 | 0.469 | bóg, dusza, miłość, światłość, wieczność |
| pos | 2 | 26 | 0.569 | 0.458 | piękny, cudny, wspaniały, cudowny, piękno |
| pos | 3 | 20 | 0.562 | 0.445 | pokazac, zrobic, zyc, zobaczyc, oddac |
| pos | 4 | 16 | 0.618 | 0.415 | pokochać, kochać, miłować, wielbić, kochany |
| neg | 1 | 27 | 0.484 | -0.540 | nagminny, pomijać, notoryczny, bagatelizować, nękać |
| neg | 2 | 10 | 0.589 | -0.473 | mimo, pomimo, wprawdzie, niestety, prawdopodobnie |
| neg | 3 | 36 | 0.627 | -0.469 | niejednokrotnie, często, nieraz, zazwyczaj, przeważnie |
| neg | 4 | 12 | 0.615 | -0.462 | głównie, głównia, również, także, zwłaszcza |
| neg | 5 | 15 | 0.745 | -0.396 | dość, niezbyt, dosyć, wyjątkowo, nazbyt |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 58 | 0.529 | 0.501 | bóg, dusza, miłość, piękno, doskonałość |
| pos | 2 | 42 | 0.527 | 0.484 | pokochać, kochać, piekne, kochany, och |
| neg | 1 | 25 | 0.492 | -0.537 | nagminny, nękać, notoryczny, bagatelizować, szykanować |
| neg | 2 | 24 | 0.526 | -0.521 | nadal, wprawdzie, wciąż, początkowo, większość |
| neg | 3 | 30 | 0.658 | -0.454 | często, niejednokrotnie, sporadycznie, nieraz, zazwyczaj |
| neg | 4 | 21 | 0.674 | -0.431 | dość, dosyć, niezbyt, wyjątkowo, nader |

## Aggregate

### SSDLite PLS
- Datasets: 7
- Significant (p < 0.05): 6/7
- R² range: 0.0999 – 0.1873 (median 0.1415)
- adj R² range: 0.0984 – 0.1859 (median 0.1399)

### SSDLite PCA+OLS
- Datasets: 7
- Significant (p < 0.05): 6/7
- R² range: 0.0489 – 0.1146 (median 0.0824)
- adj R² range: 0.0117 – 0.0827 (median 0.0465)

### Official (PCA)
- Datasets: 7
- Significant (p < 0.05): 6/7
- R² range: 0.0489 – 0.1146 (median 0.0824)
- adj R² range: 0.0117 – 0.0827 (median 0.0465)

