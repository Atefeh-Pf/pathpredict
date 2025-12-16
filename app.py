"""
PathPredict Dashboard
=====================
Streamlit Dashboard für Educational Success Forecasting

Autor: Ati
Projekt: PathPredict Capstone (neuefische Data Science Bootcamp)
"""

# Streamlit für Web-App
import streamlit as st

# Pandas für DataFrames
import pandas as pd

# NumPy für numerische Operationen
import numpy as np

# Matplotlib für Plots
import matplotlib.pyplot as plt

# Seaborn für erweiterte Visualisierungen
import seaborn as sns

# Plotly für interaktive Plots
import plotly.express as px
import plotly.graph_objects as go

# Path für Dateipfade
from pathlib import Path

# joblib zum Laden von Modellen
import joblib

# Warnings unterdrücken
import warnings
warnings.filterwarnings('ignore')

# Importiere Page-Funktionen
from dashboard_pages import (
    show_cluster_analysis,
    show_regional_comparison,
    show_model_performance,
    show_prediction_tool
)


# ============================================================
# PAGE CONFIG
# ============================================================

# Setze Seiten-Konfiguration
st.set_page_config(
    page_title="PathPredict Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

# Custom CSS für besseres Design
st.markdown("""
<style>
    /* Haupttitel-Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Metriken-Karten */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Info-Boxen */
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    """
    Lade Hauptdatensatz mit caching
    
    Returns:
    --------
    pd.DataFrame mit allen Daten
    """
    # Pfad zur Datei
    data_path = 'data/processed/soep_with_regions.csv'
    
    # Prüfe ob Datei existiert
    if not Path(data_path).exists():
        # Fehlermeldung
        st.error(f'Datei nicht gefunden: {data_path}')
        st.stop()
    
    # Lade CSV
    df = pd.read_csv(data_path)
    
    # Gib DataFrame zurück
    return df


@st.cache_data
def load_cluster_profiles():
    """Lade Cluster-Profile"""
    # Pfad zur Datei
    path = 'data/processed/cluster_profiles.csv'
    
    # Prüfe ob Datei existiert
    if not Path(path).exists():
        return None
    
    # Lade CSV
    df = pd.read_csv(path)
    
    # Gib DataFrame zurück
    return df


@st.cache_data
def load_model_comparison():
    """Lade Model-Vergleichstabelle"""
    # Pfad zur Datei
    path = 'data/processed/model_comparison.csv'
    
    # Prüfe ob Datei existiert
    if not Path(path).exists():
        return None
    
    # Lade CSV
    df = pd.read_csv(path)
    
    # Gib DataFrame zurück
    return df


@st.cache_resource
def load_best_model():
    """Lade bestes ML-Modell"""
    # Pfad zum Modell
    model_path = 'data/processed/best_model.pkl'
    
    # Prüfe ob Datei existiert
    if not Path(model_path).exists():
        return None
    
    # Lade Modell
    model = joblib.load(model_path)
    
    # Gib Modell zurück
    return model


@st.cache_resource
def load_scaler():
    """Lade Scaler für Features"""
    # Pfad zum Scaler
    scaler_path = 'data/processed/scaler.pkl'
    
    # Prüfe ob Datei existiert
    if not Path(scaler_path).exists():
        return None
    
    # Lade Scaler
    scaler = joblib.load(scaler_path)
    
    # Gib Scaler zurück
    return scaler


@st.cache_data
def load_feature_names():
    """Lade Feature-Namen"""
    # Pfad zur Datei
    path = 'data/processed/feature_names.txt'
    
    # Prüfe ob Datei existiert
    if not Path(path).exists():
        return None
    
    # Lese Zeilen
    with open(path, 'r') as f:
        # Entferne Newlines
        features = [line.strip() for line in f.readlines()]
    
    # Gib Liste zurück
    return features


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_metric_card(label, value, delta=None):
    """
    Erstelle Metriken-Karte
    
    Parameters:
    -----------
    label : str
        Label der Metrik
    value : str/float
        Wert der Metrik
    delta : str, optional
        Änderung (mit Pfeil)
    """
    # Erstelle Spalte
    col = st.columns(1)[0]
    
    # Zeige Metrik
    col.metric(
        label=label,
        value=value,
        delta=delta
    )


def format_large_number(num):
    """
    Formatiere große Zahlen mit K, M
    
    Beispiel: 1500 -> 1.5K
    """
    # Prüfe Größe
    if num >= 1_000_000:
        # Millionen
        return f'{num/1_000_000:.1f}M'
    elif num >= 1_000:
        # Tausende
        return f'{num/1_000:.1f}K'
    else:
        # Normale Zahl
        return str(int(num))


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Hauptfunktion der App"""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 PathPredict Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Educational Success Forecasting with ML & Regional Data</p>', unsafe_allow_html=True)
    
    # Trennlinie
    st.markdown('---')
    
    # Lade Daten
    with st.spinner('Lade Daten...'):
        # Hauptdaten
        df = load_data()
        
        # Cluster-Profile
        cluster_profiles = load_cluster_profiles()
        
        # Model-Vergleich
        model_comparison = load_model_comparison()
        
        # ML-Modell
        model = load_best_model()
        scaler = load_scaler()
        feature_names = load_feature_names()
    
    # Sidebar
    with st.sidebar:
        # Logo/Titel
        st.title('🎓 PathPredict')
        st.markdown('---')
        
        # Navigation
        page = st.radio(
            'Navigation',
            ['📊 Daten-Explorer', 
             '🔍 Cluster-Analyse', 
             '🗺️ Regional-Vergleich',
             '📈 Model Performance',
             '🎯 Vorhersage-Tool'],
            label_visibility='collapsed'
        )
        
        st.markdown('---')
        
        # Info-Box
        st.info('''
        **PathPredict** kombiniert individuelle SOEP-Daten mit regionalen INKAR-Statistiken 
        für präzise Bildungserfolg-Vorhersagen.
        
        **Datenquellen:**
        - SOEP (2015-2019)
        - INKAR Regional Data
        ''')
        
        # Statistiken
        st.markdown('### 📊 Datensatz')
        st.metric('Beobachtungen', format_large_number(len(df)))
        st.metric('Features', len(df.columns))
        st.metric('Jahre', f"{df['syear'].min()}-{df['syear'].max()}" if 'syear' in df.columns else 'N/A')
    
    # Page Routing
    if page == '📊 Daten-Explorer':
        # Zeige Daten-Explorer
        show_data_explorer(df)
    elif page == '🔍 Cluster-Analyse':
        # Zeige Cluster-Analyse
        show_cluster_analysis(df, cluster_profiles)
    elif page == '🗺️ Regional-Vergleich':
        # Zeige Regional-Vergleich
        show_regional_comparison(df)
    elif page == '📈 Model Performance':
        # Zeige Model Performance
        show_model_performance(model_comparison, df)
    elif page == '🎯 Vorhersage-Tool':
        # Zeige Prediction Tool
        show_prediction_tool(model, scaler, feature_names, df)


# ============================================================
# PAGE 1: DATEN-EXPLORER
# ============================================================

def show_data_explorer(df):
    """Zeige Daten-Explorer Page"""
    
    # Titel
    st.header('📊 Daten-Explorer')
    st.markdown('Erkunde den PathPredict-Datensatz interaktiv')
    
    # Tabs für verschiedene Ansichten
    tab1, tab2, tab3 = st.tabs(['📋 Übersicht', '🔍 Filter & Suche', '📈 Statistiken'])
    
    # ========== TAB 1: ÜBERSICHT ==========
    with tab1:
        # Metriken-Zeile
        col1, col2, col3, col4 = st.columns(4)
        
        # Spalte 1: Gesamt-Beobachtungen
        with col1:
            st.metric('Gesamt', format_large_number(len(df)))
        
        # Spalte 2: Durchschnittliches Einkommen
        with col2:
            if 'einkommenj1' in df.columns:
                avg_income = df['einkommenj1'].mean()
                st.metric('Ø Einkommen', f'€{avg_income:,.0f}')
        
        # Spalte 3: Durchschnittliche Bildung
        with col3:
            if 'bildung' in df.columns:
                avg_edu = df['bildung'].mean()
                st.metric('Ø Bildung', f'{avg_edu:.1f} Jahre')
        
        # Spalte 4: High Education Rate
        with col4:
            if 'high_education' in df.columns:
                high_edu_rate = df['high_education'].mean() * 100
                st.metric('High Education', f'{high_edu_rate:.1f}%')
        
        # Trennlinie
        st.markdown('---')
        
        # Daten-Tabelle
        st.subheader('📋 Daten-Vorschau')
        
        # Anzahl Zeilen auswählen
        n_rows = st.slider('Anzahl Zeilen', 5, 100, 10)
        
        # Zeige DataFrame
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        # Download-Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Daten herunterladen (CSV)',
            data=csv,
            file_name='pathpredict_data.csv',
            mime='text/csv'
        )
    
    # ========== TAB 2: FILTER & SUCHE ==========
    with tab2:
        st.subheader('🔍 Filter-Optionen')
        
        # Filter-Spalten
        col1, col2, col3 = st.columns(3)
        
        # Spalte 1: Jahr-Filter
        with col1:
            if 'syear' in df.columns:
                # Multiselect für Jahre
                years = sorted(df['syear'].unique())
                selected_years = st.multiselect(
                    'Jahr',
                    years,
                    default=years
                )
        
        # Spalte 2: Bundesland-Filter
        with col2:
            if 'matched_bundesland' in df.columns:
                # Multiselect für Bundesländer
                bundeslaender = sorted(df['matched_bundesland'].dropna().unique())
                selected_bl = st.multiselect(
                    'Bundesland',
                    bundeslaender,
                    default=bundeslaender[:3]  # Erste 3 als Default
                )
        
        # Spalte 3: Bildungs-Filter
        with col3:
            if 'high_education' in df.columns:
                # Selectbox für High Education
                edu_filter = st.selectbox(
                    'Bildungsniveau',
                    ['Alle', 'High Education (>=12 Jahre)', 'Lower Education (<12 Jahre)']
                )
        
        # Wende Filter an
        df_filtered = df.copy()
        
        # Jahr-Filter
        if 'syear' in df.columns and selected_years:
            df_filtered = df_filtered[df_filtered['syear'].isin(selected_years)]
        
        # Bundesland-Filter
        if 'matched_bundesland' in df.columns and selected_bl:
            df_filtered = df_filtered[df_filtered['matched_bundesland'].isin(selected_bl)]
        
        # Bildungs-Filter
        if 'high_education' in df.columns:
            if edu_filter == 'High Education (>=12 Jahre)':
                df_filtered = df_filtered[df_filtered['high_education'] == 1]
            elif edu_filter == 'Lower Education (<12 Jahre)':
                df_filtered = df_filtered[df_filtered['high_education'] == 0]
        
        # Info über gefilterte Daten
        st.info(f'📊 **{len(df_filtered)}** von {len(df)} Beobachtungen angezeigt')
        
        # Zeige gefilterte Daten
        st.dataframe(df_filtered.head(50), use_container_width=True)
    
    # ========== TAB 3: STATISTIKEN ==========
    with tab3:
        st.subheader('📈 Deskriptive Statistiken')
        
        # Wähle numerische Spalten
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Multiselect für Spalten
        selected_cols = st.multiselect(
            'Wähle Spalten',
            numeric_cols,
            default=numeric_cols[:5]  # Erste 5 als Default
        )
        
        if selected_cols:
            # Zeige describe()
            st.dataframe(df[selected_cols].describe(), use_container_width=True)
            
            # Trennlinie
            st.markdown('---')
            
            # Korrelations-Matrix
            st.subheader('🔗 Korrelations-Matrix')
            
            # Berechne Korrelation
            corr = df[selected_cols].corr()
            
            # Erstelle Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax)
            ax.set_title('Korrelations-Matrix')
            
            # Zeige Plot
            st.pyplot(fig)


# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':
    # Starte App
    main()
