---

### Analiza slik Edvard-a Munch-a

#### predmet Osnove oblikovanja na FRI, dodiplomski študij, april 2026

#### Avtorja: Martin Malenšek, Urša Vavpotič

---

Slike so zbrane in dostopne na [kaggle spletni strani](https://www.kaggle.com/datasets/isaienkov/edvard-munch-paintings).

---

Sorodna dela:

[Large-Scale Quantitative Analysis of Painting Arts](https://pmc.ncbi.nlm.nih.gov/articles/PMC4263068/)
- uporaba barv: kako pogosto se pojavijo specifične barve
- barvna raznolikost
- svetlost in moč kontrasta

[The Sky in Edvard Munch’s The Scream](https://journals.ametsoc.org/view/journals/bams/99/7/bams-d-17-0144.1.xml?tab_body=pdf)
- analiza vzorcev in barv neba na sliki "Krik"
---

Rezultat prototipa:
![Rezultat prototipa](./rezultati_tmp/slike_analiza.png)

---

## Zagon

Namesti odvisnosti (v venv/conda env):

```bash
python3 -m pip install -r requirements.txt
```

Zagon glavne skripte za časovno analizo (primeri):

```bash
# barva + tekstura
python3 casovna_analiza.py --start 1 --end 100 --folder ../munch_paintings --csv edvard_munch.csv --mode both

# samo barva
python3 casovna_analiza.py --start 1 --end 100 --mode color

# samo edge/tekstura
python3 casovna_analiza.py --start 1 --end 100 --mode edge
```

Opombe:
- Privzeti `--folder` je `../munch_paintings` in privzeti `--csv` je `edvard_munch.csv` 
- `--mode` je lahko `color`, `edge`, ali `both`.


Skripta analiza_barv.py je prvi Martinov prototip (da ne pozabiva).