# PathPredict Dashboard - Deployment Guide

## 📋 Übersicht

Dieses README erklärt wie das PathPredict Dashboard auf **Streamlit Cloud** deployed werden kann.

---

## 🚀 Deployment auf Streamlit Cloud

### Voraussetzungen

1. **GitHub Account** 
2. **Streamlit Cloud Account** (mit GitHub verbinden)
3. Alle Daten und Modelle trainiert (Notebooks 01-03 ausgeführt)

---

### Schritt 1: Repository-Struktur vorbereiten

Dein GitHub Repository sollte so aussehen:

```
pathpredict/
├── app.py                          # Haupt-Dashboard
├── dashboard_pages.py              # Page-Funktionen
├── requirements.txt                # Dependencies
├── data/
│   └── processed/
│       ├── soep_with_regions.csv
│       ├── cluster_profiles.csv
│       ├── model_comparison.csv
│       ├── best_model.pkl
│       ├── scaler.pkl
│       └── feature_names.txt
├── src/
│   └── regional_mapper.py         # Optional (nur wenn App es nutzt)
└── README.md
```

---

### Schritt 2: GitHub Repository erstellen

#### Option A: Via GitHub Website

1. Gehe zu https://github.com/new
2. Repository Name: `pathpredict`
3. Beschreibung: "Educational Success Forecasting Dashboard"
4. Public (für Streamlit Cloud Free)
5. Klicke "Create repository"

#### Option B: Via Command Line

```bash
# Im Projekt-Ordner
git init
git add .
git commit -m "Initial commit: PathPredict Dashboard"
git branch -M main
git remote add origin https://github.com/Atefeh-Pf/pathpredict.git
git push -u origin main
```

---

### Schritt 3: Streamlit Cloud Setup

1. **Gehe zu:** https://streamlit.io/cloud
2. **Sign in** mit GitHub
3. **New app** klicken
4. **Repository auswählen:** `DEIN-USERNAME/pathpredict`
5. **Branch:** `main`
6. **Main file path:** `app.py`
7. **Deploy!** klicken

---

### Schritt 4: URL erhalten

Nach 2-5 Minuten ist deine App live unter:

```
https://DEIN-USERNAME-pathpredict-app-RANDOM.streamlit.app
```

Diese URL kannst du teilen!

---

## 💻 Lokales Testen (vor Deployment)

### Installation

```bash
# Erstelle Virtual Environment
python -m venv venv

# Aktiviere venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Installiere Dependencies
pip install -r requirements.txt
```

### App starten

```bash
# Im Hauptverzeichnis
streamlit run app.py
```

Die App öffnet sich automatisch unter `http://localhost:8501`

---

## 📁 Datei-Größen & Git LFS

### Problem: Große Dateien

GitHub erlaubt maximal 100MB pro Datei. ML-Modelle können größer sein.

### Lösung: Git LFS (Large File Storage)

```bash
# Installiere Git LFS
# Mac:
brew install git-lfs
# Ubuntu:
sudo apt-get install git-lfs

# Initialisiere LFS
git lfs install

# Tracke große Dateien
git lfs track "*.pkl"
git lfs track "*.csv"

# Committen
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

### Alternative: Dropbox/Google Drive

Falls Dateien > 100MB:
1. Hoste Daten auf Dropbox/Google Drive
2. In `app.py` von URL laden statt lokal

```python
# Beispiel
@st.cache_data
def load_data_from_url():
    url = "https://www.dropbox.com/s/YOUR-SHARE-LINK/data.csv?dl=1"
    df = pd.read_csv(url)
    return df
```

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError"

**Lösung:** Fehlende Dependency in `requirements.txt` ergänzen

### Problem: "FileNotFoundError"

**Lösung:** Pfade relativ zur `app.py` angeben:

```python
# Falsch
df = pd.read_csv('/Users/ati/project/data/file.csv')

# Richtig
df = pd.read_csv('data/processed/file.csv')
```

### Problem: "Memory Error"

**Lösung:** 
1. Reduziere Datensatz-Größe (Sample)
2. Nutze `st.cache_data` für Daten
3. Nutze `st.cache_resource` für Modelle

### Problem: App lädt langsam

**Lösung:**
1. Caching aktivieren (`@st.cache_data`)
2. Große Plots als Bilder laden statt live generieren
3. Datensatz vorfiltern

---

## 📊 Dashboard-Features

### Aktuell implementiert:

✅ **Daten-Explorer**
- Übersicht & Statistiken
- Interaktive Filter
- CSV-Download

✅ **Cluster-Analyse**
- 16 synthetische Cluster
- Cluster → Bundesland Mapping

✅ **Regional-Vergleich**
- INKAR-Indikatoren
- Zeitreihen 2015-2019

✅ **Model Performance**
- 5 Modelle verglichen
- Metriken-Tabellen

✅ **Vorhersage-Tool**
- Input-Formular
- Echtzeit-Prediction
- Wahrscheinlichkeits-Anzeige

---

## 🎨 Customization

### Design ändern

In `app.py` unter "CUSTOM CSS":

```python
st.markdown("""
<style>
    .main-header {
        color: #YOUR-COLOR;  /* Ändere Farbe */
    }
</style>
""", unsafe_allow_html=True)
```

### Logo hinzufügen

```python
st.sidebar.image('logo.png', width=200)
```

### Mehr Pages hinzufügen

In `dashboard_pages.py` neue Funktion erstellen:

```python
def show_new_page(df):
    st.header('Neue Page')
    # Dein Code hier
```

In `app.py` Navigation erweitern:

```python
page = st.radio(
    'Navigation',
    [..., '🆕 Neue Page']
)

if page == '🆕 Neue Page':
    show_new_page(df)
```

---

## 📧 Support

Bei Fragen:
- **GitHub Issues:** https://github.com/DEIN-USERNAME/pathpredict/issues
- **Streamlit Forum:** https://discuss.streamlit.io/

---

## 📄 Lizenz

Dieses Projekt ist Teil des neuefische Data Science Bootcamps.

**Datenquellen:**
- SOEP (Sozio-oekonomisches Panel)
- INKAR (Bundesinstitut für Bau-, Stadt- und Raumforschung)


