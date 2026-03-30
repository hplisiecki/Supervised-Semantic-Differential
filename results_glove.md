# SSD Method Comparison Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding | GloVe 800d Polish (L2 + ABTT m=1) |
| Context window | ±3 tokens |
| SIF a | 0.001 |
| **SSDLite PLS** | pls |
| **SSDLite PCA+OLS** | pcaols |
| **Official (PCA)** | official |

## Per-Dataset Results

### imigrant

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 23 | 23 |
| R² | 0.3006 | 0.0746 | 0.0746 |
| adj R² | 0.2990 | 0.0250 | 0.0250 |
| p-value | 0.02917 | 0.06405 | 0.06405 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 69.2% | 69.2% | 69.2% |
| Time (s) | 0.1 | 30.4 | 62.2 |

### klimat

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 39 | 39 |
| R² | 0.1725 | 0.1032 | 0.1032 |
| adj R² | 0.1710 | 0.0365 | 0.0365 |
| p-value | 0.00025 | 0.02027 | 0.02027 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 86.3% | 86.3% | 86.3% |
| Time (s) | 0.1 | 31.7 | 75.8 |

### naukowcy

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 68 | 68 |
| R² | 0.1987 | 0.2170 | 0.2170 |
| adj R² | 0.1974 | 0.1161 | 0.1161 |
| p-value | 0.00000 | 0.00000 | 0.00000 |
| N (observations) | 648 | 648 | 648 |
| Coverage | 92.1% | 92.1% | 92.1% |
| Time (s) | 0.1 | 34.1 | 152.6 |

### polityka

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 90 | 90 |
| R² | 0.1663 | 0.2710 | 0.2710 |
| adj R² | 0.1649 | 0.1369 | 0.1369 |
| p-value | 0.00000 | 0.00000 | 0.00000 |
| N (observations) | 648 | 648 | 648 |
| Coverage | 89.5% | 89.5% | 89.5% |
| Time (s) | 0.6 | 35.3 | 86.1 |

### szczepienie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 13 | 13 |
| R² | 0.3015 | 0.1017 | 0.1017 |
| adj R² | 0.3003 | 0.0817 | 0.0817 |
| p-value | 0.00000 | 0.00000 | 0.00000 |
| N (observations) | 655 | 655 | 655 |
| Coverage | 91.1% | 91.1% | 91.1% |
| Time (s) | 0.1 | 33.6 | 76.7 |

### zaufanie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 14 | 14 |
| R² | 0.1729 | 0.0488 | 0.0488 |
| adj R² | 0.1715 | 0.0258 | 0.0258 |
| p-value | 0.00062 | 0.00948 | 0.00948 |
| N (observations) | 636 | 636 | 636 |
| Coverage | 93.5% | 93.5% | 93.5% |
| Time (s) | 0.1 | 31.7 | 75.0 |

### zdrowie

| Metric | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|--------|----------|----------|----------|
| K (components) | 1 | 27 | 27 |
| R² | 0.2237 | 0.0758 | 0.0758 |
| adj R² | 0.2223 | 0.0273 | 0.0273 |
| p-value | 0.00742 | 0.03688 | 0.03688 |
| N (observations) | 636 | 636 | 636 |
| Coverage | 85.2% | 85.2% | 85.2% |
| Time (s) | 0.1 | 32.2 | 175.6 |

## Summary

| Dataset | K_PLS | R²_PLS | adj_PLS | K_PCA+OLS | R²_PCA+OLS | adj_PCA+OLS | K_(PCA) | R²_(PCA) | adj_(PCA) |
|---|---|---|---|---|---|---|---|---|---|
| imigrant | 1 | 0.3006 | 0.2990 | 23 | 0.0746 | 0.0250 | 23 | 0.0746 | 0.0250 |
| klimat | 1 | 0.1725 | 0.1710 | 39 | 0.1032 | 0.0365 | 39 | 0.1032 | 0.0365 |
| naukowcy | 1 | 0.1987 | 0.1974 | 68 | 0.2170 | 0.1161 | 68 | 0.2170 | 0.1161 |
| polityka | 1 | 0.1663 | 0.1649 | 90 | 0.2710 | 0.1369 | 90 | 0.2710 | 0.1369 |
| szczepienie | 1 | 0.3015 | 0.3003 | 13 | 0.1017 | 0.0817 | 13 | 0.1017 | 0.0817 |
| zaufanie | 1 | 0.1729 | 0.1715 | 14 | 0.0488 | 0.0258 | 14 | 0.0488 | 0.0258 |
| zdrowie | 1 | 0.2237 | 0.2223 | 27 | 0.0758 | 0.0273 | 27 | 0.0758 | 0.0273 |

## Top Words Comparison

### imigrant

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | zatrudnienie | opieka | opieka |
| 2 | podobne | edukacyjny | edukacyjny |
| 3 | emocjonalny | społeczny | społeczny |
| 4 | konieczny | psychologiczny | psychologiczny |
| 5 | niezbędny | wychowanie | wychowanie |
| 6 | twórczy | wychowawczy | wychowawczy |
| 7 | tworzenie | zaspokojenie | zaspokojenie |
| 8 | codzienny | zdrowotny | zdrowotny |
| 9 | autorski | zawód | zawód |
| 10 | proces | potrzeb | potrzeb |
| 11 | umożliwiać | niepełnosprawny | niepełnosprawny |
| 12 | obiektywny | rehabilitacja | rehabilitacja |
| 13 | warunek | znalezienie | znalezienie |
| 14 | swoisty | nadzór | nadzór |
| 15 | potrzeb | socjalna | socjalna |
| 16 | potrzeba | zatrudnienie | zatrudnienie |
| 17 | podmiot | rodzinny | rodzinny |
| 18 | pisanie | psychiczny | psychiczny |
| 19 | poznawczy | dydaktyczny | dydaktyczny |
| 20 | wykorzystanie | motywacja | motywacja |

All-way overlap: 5%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | okrążyć | anglicy | anglicy |
| 2 | egipt | egipt | egipt |
| 3 | wedrzeć | półwysep | półwysep |
| 4 | przedrzeć | wyprzeć | wyprzeć |
| 5 | wtargnąć | południowo | południowo |
| 6 | odchylić | rosjanie | rosjanie |
| 7 | najeźdźca | wschód | wschód |
| 8 | uderzyć | północno | północno |
| 9 | zdołać | najeźdźca | najeźdźca |
| 10 | zaatakować | wschodni | wschodni |
| 11 | wschód | rosja | rosja |
| 12 | powalić | północ | północ |
| 13 | nacierać | zachodni | zachodni |
| 14 | wyprzeć | północny | północny |
| 15 | pokonany | podbić | podbić |
| 16 | azji | nacierać | nacierać |
| 17 | zuchwały | zaatakować | zaatakować |
| 18 | wystrzelić | arabski | arabski |
| 19 | rebeliant | rzeka | rzeka |
| 20 | groźny | cieśnina | cieśnina |

All-way overlap: 18%

### klimat

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | upał | zmniejszenie | zmniejszenie |
| 2 | narastający | utrata | utrata |
| 3 | nadciągać | zwiększenie | zwiększenie |
| 4 | zmęczenie | przerażać | przerażać |
| 5 | odczuwać | grozić | grozić |
| 6 | niepokój | zwiększyć | zwiększyć |
| 7 | deszcz | gotowość | gotowość |
| 8 | sztorm | długotrwały | długotrwały |
| 9 | narastać | gwarantować | gwarantować |
| 10 | poczuć | wzrastać | wzrastać |
| 11 | głód | zwiększać | zwiększać |
| 12 | grozić | zmniejszyć | zmniejszyć |
| 13 | mdłości | opuszczenie | opuszczenie |
| 14 | narażać | utrzymanie | utrzymanie |
| 15 | dokuczać | osłabienie | osłabienie |
| 16 | chłód | głód | głód |
| 17 | wilgoć | obniżyć | obniżyć |
| 18 | odczuć | odczuwać | odczuwać |
| 19 | panika | brak | brak |
| 20 | przypływ | zapewnić | zapewnić |

All-way overlap: 8%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | filozofia | wytwór | wytwór |
| 2 | sa | propaganda | propaganda |
| 3 | cykl | twór | twór |
| 4 | pogląd | fantastyczny | fantastyczny |
| 5 | teza | dzieło | dzieło |
| 6 | propaganda | literacki | literacki |
| 7 | interpretacja | poglądy | poglądy |
| 8 | autorstwo | fikcyjny | fikcyjny |
| 9 | zbiór | mit | mit |
| 10 | portret | inspirować | inspirować |
| 11 | dzieło | twórczość | twórczość |
| 12 | biografia | ożywiony | ożywiony |
| 13 | teoria | wątek | wątek |
| 14 | zajmujący | autorstwo | autorstwo |
| 15 | twierdzenie | temat | temat |
| 16 | poglądy | wymyślić | wymyślić |
| 17 | twórczość | wpływowy | wpływowy |
| 18 | historyk | poetycki | poetycki |
| 19 | kanon | motyw | motyw |
| 20 | wytwór | ideologia | ideologia |

All-way overlap: 18%

### naukowcy

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | zaangażowanie | świadectwo | świadectwo |
| 2 | szacunek | artykuł | artykuł |
| 3 | osiągnięcie | ufność | ufność |
| 4 | świadectwo | szacunek | szacunek |
| 5 | ufność | mądrość | mądrość |
| 6 | moralny | dowód | dowód |
| 7 | wkład | pokora | pokora |
| 8 | twórczy | cierpliwość | cierpliwość |
| 9 | dążenie | niewinność | niewinność |
| 10 | wiara | podziw | podziw |
| 11 | rozwój | wiara | wiara |
| 12 | zapewnienie | gwarancja | gwarancja |
| 13 | zasług | zadziwiający | zadziwiający |
| 14 | uznanie | zapewnienie | zapewnienie |
| 15 | postęp | potęga | potęga |
| 16 | duchowy | powaga | powaga |
| 17 | zaufanie | bogactwo | bogactwo |
| 18 | zdolności | niebiosa | niebiosa |
| 19 | materialny | odwaga | odwaga |
| 20 | troska | milczenie | milczenie |

All-way overlap: 14%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | cholera | kierownica | kierownica |
| 2 | kurwa | wynaleźć | wynaleźć |
| 3 | okropnie | przewodzić | przewodzić |
| 4 | dzieciak | zlikwidować | zlikwidować |
| 5 | pogadać | podłączyć | podłączyć |
| 6 | dzwonić | końcówka | końcówka |
| 7 | strasznie | praktycznie | praktycznie |
| 8 | sukinsyn | zmodyfikować | zmodyfikować |
| 9 | skłamać | cholera | cholera |
| 10 | pewnie | wtedy | wtedy |
| 11 | zgadnąć | wyeliminować | wyeliminować |
| 12 | śmierdzieć | wirus | wirus |
| 13 | ukraść | wykręcić | wykręcić |
| 14 | niedobrze | przejmować | przejmować |
| 15 | oszaleć | układy | układy |
| 16 | spóźnić | wylecieć | wylecieć |
| 17 | denerwować | wyprzeć | wyprzeć |
| 18 | zabawić | kurwa | kurwa |
| 19 | wleźć | czujnik | czujnik |
| 20 | facet | znienawidzić | znienawidzić |

All-way overlap: 5%

### polityka

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | parlamentarny | przyczyniać | przyczyniać |
| 2 | wybory | socjalny | socjalny |
| 3 | wyborczy | zróżnicowanie | zróżnicowanie |
| 4 | samorządowy | powszechny | powszechny |
| 5 | głosowanie | omówić | omówić |
| 6 | parlament | świadczenie | świadczenie |
| 7 | prezydencki | wynik | wynik |
| 8 | obywatelski | kształtować | kształtować |
| 9 | uczestnictwo | dzięki | dzięki |
| 10 | samorząd | obniżenie | obniżenie |
| 11 | mandat | ćwiczenie | ćwiczenie |
| 12 | kadencja | oświata | oświata |
| 13 | konstytucyjny | dyskusja | dyskusja |
| 14 | sejm | przystosowanie | przystosowanie |
| 15 | poprzez | prawidłowy | prawidłowy |
| 16 | uchwalić | szkolnictwo | szkolnictwo |
| 17 | ubiegać | modlitwa | modlitwa |
| 18 | senat | podwyższenie | podwyższenie |
| 19 | demokratyczny | mieszkaniowy | mieszkaniowy |
| 20 | członkowski | forum | forum |

All-way overlap: 0%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | naprawdę | podejrzany | podejrzany |
| 2 | dureń | dureń | dureń |
| 3 | nic | głupiec | głupiec |
| 4 | głupiec | szpieg | szpieg |
| 5 | interesować | podejrzewać | podejrzewać |
| 6 | chyba | podejrzana | podejrzana |
| 7 | czyżby | żaden | żaden |
| 8 | cóż | podróżnik | podróżnik |
| 9 | kompletnie | intruz | intruz |
| 10 | wiedzieć | nikt | nikt |
| 11 | głupi | szaleniec | szaleniec |
| 12 | pieprzyć | idiota | idiota |
| 13 | domyślać | znać | znać |
| 14 | ależ | banda | banda |
| 15 | burknąć | morderca | morderca |
| 16 | podejrzewać | prócz | prócz |
| 17 | nikt | niebezpieczny | niebezpieczny |
| 18 | idiota | nic | nic |
| 19 | doprawdy | dziwka | dziwka |
| 20 | wariat | interesować | interesować |

All-way overlap: 21%

### szczepienie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | postęp | przyczyniać | przyczyniać |
| 2 | rozwój | rozwój | rozwój |
| 3 | przyczyniać | propagować | propagować |
| 4 | szerzyć | postęp | postęp |
| 5 | wkład | edukacja | edukacja |
| 6 | propagować | ludzkość | ludzkość |
| 7 | sprzyjać | szerzyć | szerzyć |
| 8 | ludzkość | umacniać | umacniać |
| 9 | szczególnie | sprzyjać | sprzyjać |
| 10 | rozwijać | społeczeństwo | społeczeństwo |
| 11 | rozwijająca | duchowy | duchowy |
| 12 | przejaw | integracja | integracja |
| 13 | globalny | kulturalny | kulturalny |
| 14 | duchowy | oświata | oświata |
| 15 | istotny | ważny | ważny |
| 16 | osiągnięcie | społeczny | społeczny |
| 17 | zaleta | oświecenie | oświecenie |
| 18 | kluczowy | kluczowy | kluczowy |
| 19 | dzięki | wspomagać | wspomagać |
| 20 | oświecenie | idea | idea |

All-way overlap: 33%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | dobrowolnie | powtórnie | powtórnie |
| 2 | odjechać | odczekać | odczekać |
| 3 | przyjechać | zgłosić | zgłosić |
| 4 | zamieszkać | sprawdzić | sprawdzić |
| 5 | przebrać | przeszukać | przeszukać |
| 6 | posłusznie | zarządzić | zarządzić |
| 7 | powtórnie | uchylić | uchylić |
| 8 | niechętnie | zbadać | zbadać |
| 9 | materac | zdecydować | zdecydować |
| 10 | nazajutrz | obejrzeć | obejrzeć |
| 11 | przyjeżdżać | odmówić | odmówić |
| 12 | pojechać | orzec | orzec |
| 13 | rano | zażądać | zażądać |
| 14 | strych | wpuścić | wpuścić |
| 15 | zgadnąć | zgodzić | zgodzić |
| 16 | pogadać | nazajutrz | nazajutrz |
| 17 | spać | wycofać | wycofać |
| 18 | zabawić | przewieźć | przewieźć |
| 19 | ochotnik | zamówić | zamówić |
| 20 | ano | poinformować | poinformować |

All-way overlap: 5%

### zaufanie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | zróżnicować | głównie | głównie |
| 2 | charakterystyka | popularny | popularny |
| 3 | ogólnopolski | interesujący | interesujący |
| 4 | sąsiadować | zajmujący | zajmujący |
| 5 | charakteryzować | przeważnie | przeważnie |
| 6 | zbliżony | tematyka | tematyka |
| 7 | popularny | publikować | publikować |
| 8 | głównie | nawiązywać | nawiązywać |
| 9 | xx | szczególnie | szczególnie |
| 10 | kulturalny | ciekawostka | ciekawostka |
| 11 | uprawa | prezentować | prezentować |
| 12 | xix | modny | modny |
| 13 | ozdobny | liczny | liczny |
| 14 | występowanie | kulturalny | kulturalny |
| 15 | stosunkowo | wymiana | wymiana |
| 16 | zwany | prowadzony | prowadzony |
| 17 | rozwijająca | różnorodny | różnorodny |
| 18 | międzywojenny | organizować | organizować |
| 19 | ciekawostka | atrakcyjny | atrakcyjny |
| 20 | porównanie | charakterystyka | charakterystyka |

All-way overlap: 14%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | patrzyć | strach | strach |
| 2 | zawahać | trwoga | trwoga |
| 3 | unieść | poczuć | poczuć |
| 4 | milczeć | zawahać | zawahać |
| 5 | spojrzeć | przerażenie | przerażenie |
| 6 | zaufać | lęk | lęk |
| 7 | wściekłość | wyczuć | wyczuć |
| 8 | złość | wściekłość | wściekłość |
| 9 | przemówić | zadrżeć | zadrżeć |
| 10 | przyglądać | drżeć | drżeć |
| 11 | zdradzić | ostrożność | ostrożność |
| 12 | zrozumieć | bać | bać |
| 13 | uwierzyć | zacisnąć | zacisnąć |
| 14 | krzyknąć | niepokój | niepokój |
| 15 | czuć | instynktownie | instynktownie |
| 16 | trwoga | złość | złość |
| 17 | domyślić | gniew | gniew |
| 18 | ufać | szepnąć | szepnąć |
| 19 | odwrócić | krzyknąć | krzyknąć |
| 20 | krzyczeć | zdenerwowanie | zdenerwowanie |

All-way overlap: 14%

### zdrowie

**Positive pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | harmonia | zależeć | zależeć |
| 2 | porządek | dziękować | dziękować |
| 3 | wspaniale | harmonia | harmonia |
| 4 | zadowolenie | spokój | spokój |
| 5 | zadbać | zapewnić | zapewnić |
| 6 | doskonały | oby | oby |
| 7 | zapewnić | aha | aha |
| 8 | dbać | doskonałość | doskonałość |
| 9 | ułożyć | kochanie | kochanie |
| 10 | popatrzeć | kochana | kochana |
| 11 | idealny | równowaga | równowaga |
| 12 | potrzebny | życzyć | życzyć |
| 13 | śliczny | doskonały | doskonały |
| 14 | podziw | głaskać | głaskać |
| 15 | zgrabny | względny | względny |
| 16 | pochwalić | idealny | idealny |
| 17 | stworzyć | porządek | porządek |
| 18 | porządny | ciepło | ciepło |
| 19 | satysfakcja | szczęście | szczęście |
| 20 | póki | intuicja | intuicja |

All-way overlap: 14%

**Negative pole:**

| # | SSDLite PLS | SSDLite PCA+OLS | Official (PCA) |
|---||-------|-------|-------|
| 1 | obecnie | wskutek | wskutek |
| 2 | wskutek | skazany | skazany |
| 3 | fatalny | okupacja | okupacja |
| 4 | częsty | aresztowany | aresztowany |
| 5 | powód | zniszczony | zniszczony |
| 6 | wyczerpany | oskarżać | oskarżać |
| 7 | niedawny | śmiertelnie | śmiertelnie |
| 8 | niekorzystny | niekorzystny | niekorzystny |
| 9 | śmiertelnie | przygnębiony | przygnębiony |
| 10 | krytykować | powód | powód |
| 11 | osłabić | wyczerpany | wyczerpany |
| 12 | szacować | zły | zły |
| 13 | przygnębiony | międzywojenny | międzywojenny |
| 14 | niestety | krytykować | krytykować |
| 15 | poważnie | likwidacja | likwidacja |
| 16 | załamać | wojenny | wojenny |
| 17 | skutek | represja | represja |
| 18 | doniesienie | obecnie | obecnie |
| 19 | zgon | ówczesny | ówczesny |
| 20 | oskarżać | straszliwie | straszliwie |

All-way overlap: 29%

## Clusters Comparison

### imigrant

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 62 | 0.515 | 0.370 | potrzeba, dotyczyć, niezbędny, podobne, praca |
| pos | 2 | 38 | 0.532 | 0.360 | emocjonalny, psychiczny, określony, poznawczy, wynikać |
| neg | 1 | 67 | 0.385 | -0.420 | zaatakować, uderzyć, najeźdźca, zdołać, wedrzeć |
| neg | 2 | 33 | 0.539 | -0.313 | iran, turcja, egipt, syria, afganistan |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 45 | 0.508 | 0.424 | potrzeba, potrzeb, społeczny, psychiczny, emocjonalny |
| pos | 2 | 55 | 0.526 | 0.414 | kształcenie, pedagogiczny, edukacja, wychowawczy, wychowanie |
| neg | 1 | 32 | 0.506 | -0.395 | anglicy, turcy, rosjanie, szwedzi, brytyjczycy |
| neg | 2 | 29 | 0.564 | -0.366 | iran, afganistan, irak, egipt, pakistan |
| neg | 3 | 15 | 0.534 | -0.349 | zaatakować, uderzyć, zniszczyć, osłabić, wycofywać |
| neg | 4 | 24 | 0.602 | -0.330 | północny, południowo, południowy, wschodni, zachodni |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 72 | 0.500 | 0.433 | opieka, kształcenie, wychowawczy, edukacja, wychowanie |
| pos | 2 | 28 | 0.574 | 0.378 | emocjonalny, moralny, potrzeba, psychiczny, potrzeb |
| neg | 1 | 21 | 0.497 | -0.396 | rosję, turcję, syrię, azję, japonię |
| neg | 2 | 24 | 0.561 | -0.352 | anglicy, rosjanie, brytyjczycy, turcy, japończycy |
| neg | 3 | 23 | 0.585 | -0.348 | iran, afganistan, irak, turcja, pakistan |
| neg | 4 | 11 | 0.577 | -0.347 | najechać, pustoszyć, najeżdżać, spustoszyć, zagarnąć |
| neg | 5 | 21 | 0.627 | -0.319 | północny, południowo, południowy, wschodni, zachodni |

### klimat

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 50 | 0.496 | 0.422 | odczuwać, ból, odczuć, czuć, poczuć |
| pos | 2 | 22 | 0.573 | 0.371 | deszcz, ulew, ulewny, ulewa, upał |
| pos | 3 | 23 | 0.593 | 0.351 | niepokój, narastający, strach, gniew, złość |
| pos | 4 | 5 | 0.618 | 0.350 | dyskomfort, niewygoda, nerwowość, uskarżać, podenerwować |
| neg | 1 | 15 | 0.431 | -0.446 | ideologia, propaganda, masoński, masoneria, fikcja |
| neg | 2 | 85 | 0.494 | -0.383 | pogląd, autor, teoria, filozofia, współczesny |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 44 | 0.485 | 0.378 | odczuwać, niepokój, poczucie, zmęczenie, głód |
| pos | 2 | 56 | 0.507 | 0.368 | zwiększenie, zmniejszenie, konieczność, zwiększać, zwiększyć |
| neg | 1 | 23 | 0.431 | -0.476 | wywrotowy, propaganda, rozpowszechniać, demaskować, antypolski |
| neg | 2 | 26 | 0.481 | -0.432 | historyjka, zmyślić, anegdota, wymyślić, opowieść |
| neg | 3 | 51 | 0.490 | -0.404 | literacki, twórczość, literatura, inspirować, temat |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 40 | 0.514 | 0.356 | odczuwać, niepokój, poczucie, głód, zmęczenie |
| pos | 2 | 25 | 0.518 | 0.349 | zapewnienie, konieczność, zapewnić, utrzymanie, gwarantować |
| pos | 3 | 30 | 0.556 | 0.345 | zmniejszenie, zwiększać, zwiększenie, zmniejszać, zwiększyć |
| pos | 4 | 5 | 0.674 | 0.268 | narażać, ryzykować, narażenie, niewygoda, dyskomfort |
| neg | 1 | 38 | 0.445 | -0.467 | zmyślić, wymyślić, historyjka, wymysł, opowieść |
| neg | 2 | 62 | 0.450 | -0.441 | literacki, twórczość, literatura, inspirować, temat |

### naukowcy

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 12 | 0.613 | 0.441 | szacunek, poszanowanie, zaufanie, troska, zapewnienie |
| pos | 2 | 25 | 0.594 | 0.437 | dziedzina, rozwój, nauka, naukowy, osiągnięcie |
| pos | 3 | 27 | 0.598 | 0.428 | szczególny, duchowy, moralny, wyrażać, dążenie |
| pos | 4 | 15 | 0.602 | 0.418 | męstwo, wiara, wytrwałość, cnota, poświęcenie |
| pos | 5 | 21 | 0.609 | 0.413 | możliwości, niezbędny, wiedza, ocena, doświadczenie |
| neg | 1 | 54 | 0.376 | -0.492 | pewnie, cholera, sukinsyn, skurwiel, dzieciak |
| neg | 2 | 46 | 0.390 | -0.468 | strasznie, okropnie, wściekać, poskarżyć, wygłupiać |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 73 | 0.446 | 0.401 | niezwykły, pełny, mądrość, bogactwo, godny |
| pos | 2 | 27 | 0.575 | 0.301 | życzliwość, odwaga, łagodność, szlachetność, cierpliwość |
| neg | 1 | 56 | 0.288 | -0.559 | pokłócić, połapać, popsuć, oberwać, niechcący |
| neg | 2 | 44 | 0.393 | -0.415 | sterowany, sterujący, montować, podłączyć, obudowa |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 66 | 0.448 | 0.396 | niezwykły, bogactwo, godny, pełny, mądrość |
| pos | 2 | 34 | 0.552 | 0.321 | życzliwość, łagodność, szczerość, szlachetność, odwaga |
| neg | 1 | 41 | 0.338 | -0.485 | oberwać, połapać, nawalić, popsuć, pokłócić |
| neg | 2 | 59 | 0.331 | -0.484 | sterowany, montować, sterujący, podłączyć, sterować |

### polityka

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 43 | 0.572 | 0.470 | wybory, parlamentarny, parlament, wyborczy, głosowanie |
| pos | 2 | 43 | 0.542 | 0.449 | komisja, komitet, przewodniczący, powołany, przewodniczyć |
| pos | 3 | 14 | 0.618 | 0.383 | zwiększenie, umożliwić, umożliwiać, wprowadzenie, zmniejszenie |
| neg | 1 | 39 | 0.509 | -0.490 | idiota, cholerny, dureń, głupi, głupiec |
| neg | 2 | 61 | 0.613 | -0.409 | naprawdę, przecież, chyba, pewno, myśleć |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 47 | 0.283 | 0.496 | dyskusja, dzięki, modlitwa, spontanicznie, debata |
| pos | 2 | 53 | 0.356 | 0.402 | przyczyniać, socjalny, kształtować, obniżenie, efektywność |
| neg | 1 | 55 | 0.435 | -0.378 | nikt, nic, żaden, podejrzewać, znaczyć |
| neg | 2 | 45 | 0.489 | -0.335 | łajdak, oszust, banda, morderca, złodziej |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 45 | 0.296 | 0.473 | oświata, szkolnictwo, mieszkaniowy, powszechny, pańszczyzna |
| pos | 2 | 55 | 0.338 | 0.425 | kształtować, przyczyniać, prawidłowy, przystosowanie, dzięki |
| neg | 1 | 12 | 0.466 | -0.353 | bezwartościowy, bezużyteczny, niewart, kompletnie, błyskotka |
| neg | 2 | 44 | 0.501 | -0.331 | łajdak, morderca, złodziej, oszust, łotr |
| neg | 3 | 30 | 0.547 | -0.299 | nikt, nic, żaden, podejrzewać, tajemniczy |
| neg | 4 | 9 | 0.541 | -0.294 | zazdrosna, zazdrosny, posądzać, szpiegować, podejrzliwy |
| neg | 5 | 5 | 0.563 | -0.288 | uroić, zaczepienie, nierealny, zerowy, poszlak |

### szczepienie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 70 | 0.533 | 0.419 | istotny, zwłaszcza, sprzyjać, rozwój, przyczyniać |
| pos | 2 | 30 | 0.532 | 0.410 | propagować, upowszechniać, społeczny, idea, edukacja |
| neg | 1 | 65 | 0.351 | -0.417 | przyjechać, pojechać, pogadać, rano, zadzwonić |
| neg | 2 | 13 | 0.369 | -0.382 | podstawić, przetestować, doświadczalnie, losowo, przeliczyć |
| neg | 3 | 11 | 0.423 | -0.343 | materac, strych, wyleźć, wczołgać, posłusznie |
| neg | 4 | 11 | 0.473 | -0.309 | nielegalnie, legalnie, przymusowo, dobrowolnie, samowolnie |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 55 | 0.504 | 0.519 | istotny, społeczeństwo, idea, dążenie, sprzyjać |
| pos | 2 | 45 | 0.529 | 0.504 | rozwój, społeczny, edukacja, kultura, upowszechniać |
| neg | 1 | 44 | 0.459 | -0.408 | sprawdzić, zbadać, obejrzeć, wyjąć, upewnić |
| neg | 2 | 56 | 0.489 | -0.380 | zdecydować, postanowić, poprosić, natychmiast, poinformować |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 23 | 0.556 | 0.499 | upowszechniać, edukacja, propagować, popularyzacja, oświata |
| pos | 2 | 40 | 0.535 | 0.480 | istotny, sprzyjać, przyczyniać, rozwój, szczególnie |
| pos | 3 | 35 | 0.555 | 0.474 | społeczeństwo, idea, społeczny, filozofia, cywilizacja |
| pos | 4 | 2 | 0.860 | 0.308 | orędownik, propagator |
| neg | 1 | 15 | 0.519 | -0.376 | przesłuchanie, areszt, przesłuchiwać, przesłuchać, zarządzić |
| neg | 2 | 32 | 0.498 | -0.370 | sprawdzić, zbadać, obejrzeć, wyjąć, dokładnie |
| neg | 3 | 28 | 0.513 | -0.366 | zgodzić, zdecydować, postanowić, odmówić, poinformować |
| neg | 4 | 25 | 0.579 | -0.316 | rano, pojechać, zadzwonić, zdążyć, poprosić |

### zaufanie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 69 | 0.488 | 0.378 | głównie, charakteryzować, zróżnicować, stosunkowo, dominować |
| pos | 2 | 31 | 0.492 | 0.362 | kulturalny, stowarzyszenie, kultura, edukacyjny, ogólnopolski |
| neg | 1 | 43 | 0.601 | -0.381 | spojrzeć, odwrócić, patrzyć, popatrzyć, spoglądać |
| neg | 2 | 36 | 0.617 | -0.369 | gdyby, wiedzieć, nikt, zrozumieć, żeby |
| neg | 3 | 5 | 0.612 | -0.354 | przytomność, ocknąć, ostrożność, nadludzki, zwątpić |
| neg | 4 | 16 | 0.650 | -0.351 | wściekłość, złość, złościć, gniew, krzyknąć |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 63 | 0.490 | 0.422 | organizować, działalność, kulturalny, prowadzony, stowarzyszenie |
| pos | 2 | 37 | 0.535 | 0.415 | głównie, szczególnie, liczny, popularny, interesujący |
| neg | 1 | 53 | 0.551 | -0.495 | poczuć, zacisnąć, unieść, czuć, bać |
| neg | 2 | 47 | 0.603 | -0.464 | strach, przerażenie, lęk, niepokój, wściekłość |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 41 | 0.513 | 0.422 | głównie, szczególnie, liczny, popularny, stosunkowo |
| pos | 2 | 59 | 0.505 | 0.414 | organizować, działalność, kulturalny, stowarzyszenie, prowadzony |
| neg | 1 | 45 | 0.570 | -0.479 | poczuć, zacisnąć, unieść, cofnąć, krzyknąć |
| neg | 2 | 52 | 0.591 | -0.474 | strach, przerażenie, lęk, niepokój, wściekłość |
| neg | 3 | 3 | 0.669 | -0.373 | zmrużyć, ukłucie, lirael |

### zdrowie

**SSDLite PLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 22 | 0.418 | 0.382 | przyjemność, zadowolenie, satysfakcja, podziw, rozkoszny |
| pos | 2 | 37 | 0.451 | 0.365 | doskonały, idealny, cudowny, wspaniale, wdzięk |
| pos | 3 | 25 | 0.501 | 0.331 | zapewnić, potrzebny, dbać, zadbać, spokój |
| pos | 4 | 10 | 0.497 | 0.328 | wymyśleć, wykombinować, zmyślny, obmyślić, fajny |
| pos | 5 | 6 | 0.539 | 0.285 | optymalny, maksymalizacja, samorealizacja, spożytkować, wypracować |
| neg | 1 | 31 | 0.445 | -0.374 | powszechnie, bywać, współcześnie, błędnie, obecnie |
| neg | 2 | 69 | 0.436 | -0.373 | powód, skutek, wskutek, spowodować, przyczyna |

**SSDLite PCA+OLS:**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 56 | 0.462 | 0.404 | spokój, zapewniać, doskonały, szczęście, zapewnić |
| pos | 2 | 31 | 0.495 | 0.360 | dziękować, kochanie, och, porozmawiać, niech |
| pos | 3 | 13 | 0.549 | 0.325 | pośladek, biodro, brzuch, głaskać, jedwabisty |
| neg | 1 | 57 | 0.425 | -0.483 | zły, skrajnie, tragiczny, straszliwie, wyczerpany |
| neg | 2 | 43 | 0.512 | -0.395 | wojna, zniszczony, okupacja, wojenny, powód |

**Official (PCA):**

| Side | Rank | Size | Coherence | cos(β) | Top words |
|------|------|------|-----------|--------|-----------|
| pos | 1 | 30 | 0.472 | 0.396 | prostota, wdzięk, doskonałość, doskonały, harmonia |
| pos | 2 | 35 | 0.473 | 0.377 | dziękować, kochanie, och, niech, porozmawiać |
| pos | 3 | 16 | 0.498 | 0.362 | spokój, szczęście, odpoczynek, brzuch, potrzebny |
| pos | 4 | 19 | 0.569 | 0.328 | gwarantować, zapewnienie, zapewniać, zapewnić, zagwarantować |
| neg | 1 | 56 | 0.425 | -0.485 | zły, skrajnie, tragiczny, straszliwie, wyczerpany |
| neg | 2 | 44 | 0.511 | -0.395 | wojna, zniszczony, okupacja, powód, spowodować |

## Aggregate

### SSDLite PLS
- Datasets: 7
- Significant (p < 0.05): 7/7
- R² range: 0.1663 – 0.3015 (median 0.1987)
- adj R² range: 0.1649 – 0.3003 (median 0.1974)

### SSDLite PCA+OLS
- Datasets: 7
- Significant (p < 0.05): 6/7
- R² range: 0.0488 – 0.2710 (median 0.1017)
- adj R² range: 0.0250 – 0.1369 (median 0.0365)

### Official (PCA)
- Datasets: 7
- Significant (p < 0.05): 6/7
- R² range: 0.0488 – 0.2710 (median 0.1017)
- adj R² range: 0.0250 – 0.1369 (median 0.0365)

