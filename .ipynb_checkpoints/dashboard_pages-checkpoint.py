"""
Dashboard Pages for PathPredict
================================
Enthält alle Page-Funktionen für das Streamlit Dashboard
"""

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE 2: CLUSTER-ANALYSE
# ============================================================

def show_cluster_analysis(df, cluster_profiles):
    """Zeige Cluster-Analyse Page"""
    
    # Titel
    st.header('🔍 Cluster-Analyse')
    st.markdown('16 synthetische regionale Cluster aus SOEP-Daten')
    
    # Prüfe ob cluster_profiles vorhanden
    if cluster_profiles is None:
        st.warning('Cluster-Profile nicht gefunden. Bitte Notebook 02 ausführen.')
        return
    
    # Tabs
    tab1, tab2 = st.tabs(['📊 Cluster-Übersicht', '🗺️ Cluster-Mapping'])
    
    # ========== TAB 1: ÜBERSICHT ==========
    with tab1:
        # Cluster-Größen
        if 'region_cluster' in df.columns:
            st.subheader('Cluster-Größen')
            
            # Zähle pro Cluster
            cluster_counts = df['region_cluster'].value_counts().sort_index()
            
            # Erstelle Balkendiagramm
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster ID', 'y': 'Anzahl'},
                title='Verteilung der Beobachtungen pro Cluster'
            )
            
            # Zeige Plot
            st.plotly_chart(fig, use_container_width=True)
        
        # Trennlinie
        st.markdown('---')
        
        # Cluster-Profile Tabelle
        st.subheader('Cluster-Profile (Durchschnittswerte)')
        st.dataframe(cluster_profiles, use_container_width=True)
    
    # ========== TAB 2: MAPPING ==========
    with tab2:
        st.subheader('🗺️ Cluster → Bundesland Zuordnung')
        
        # Prüfe ob matched_bundesland existiert
        if 'matched_bundesland' in df.columns and 'region_cluster' in df.columns:
            # Erstelle Mapping-Tabelle
            mapping = df[['region_cluster', 'matched_bundesland']].drop_duplicates().sort_values('region_cluster')
            
            # Zeige Tabelle
            st.dataframe(mapping, use_container_width=True)
        else:
            st.warning('Mapping-Daten nicht gefunden.')


# ============================================================
# PAGE 3: REGIONAL-VERGLEICH
# ============================================================

def show_regional_comparison(df):
    """Zeige Regional-Vergleich Page"""
    
    # Titel
    st.header('🗺️ Regional-Vergleich')
    st.markdown('Vergleiche Bundesländer anhand von INKAR-Indikatoren')
    
    # Prüfe ob Spalten existieren
    if 'matched_bundesland' not in df.columns:
        st.warning('Keine Bundesland-Daten gefunden.')
        return
    
    # Tabs
    tab1, tab2 = st.tabs(['📊 Bundesland-Vergleich', '📈 Zeitreihen'])
    
    # ========== TAB 1: VERGLEICH ==========
    with tab1:
        # Wähle Indikator
        indicators = [
            'arbeitslosenquote',
            'kinderarmut',
            'abiturquote',
            'betreuungsquote',
            'medianeinkommen'
        ]
        
        # Filtere vorhandene
        available_indicators = [i for i in indicators if i in df.columns]
        
        if available_indicators:
            # Selectbox
            selected_indicator = st.selectbox(
                'Wähle Indikator',
                available_indicators
            )
            
            # Aggregiere nach Bundesland (Durchschnitt über alle Jahre)
            df_agg = df.groupby('matched_bundesland')[selected_indicator].mean().sort_values(ascending=False)
            
            # Erstelle Balkendiagramm
            fig = px.bar(
                x=df_agg.values,
                y=df_agg.index,
                orientation='h',
                labels={'x': selected_indicator, 'y': 'Bundesland'},
                title=f'{selected_indicator} nach Bundesland (Durchschnitt)'
            )
            
            # Zeige Plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('Keine INKAR-Indikatoren gefunden.')
    
    # ========== TAB 2: ZEITREIHEN ==========
    with tab2:
        # Prüfe ob syear existiert
        if 'syear' not in df.columns:
            st.warning('Keine Jahr-Spalte (syear) gefunden.')
            return
        
        st.subheader('📈 Zeitreihen 2015-2019')
        
        # Wähle Indikator
        if available_indicators:
            selected_indicator_ts = st.selectbox(
                'Wähle Indikator für Zeitreihe',
                available_indicators,
                key='ts_indicator'
            )
            
            # Wähle Bundesländer
            bundeslaender = sorted(df['matched_bundesland'].unique())
            selected_bl = st.multiselect(
                'Wähle Bundesländer',
                bundeslaender,
                default=bundeslaender[:5]  # Erste 5
            )
            
            if selected_bl:
                # Filtere Daten
                df_ts = df[df['matched_bundesland'].isin(selected_bl)]
                
                # Aggregiere nach Jahr und Bundesland
                df_ts_agg = df_ts.groupby(['syear', 'matched_bundesland'])[selected_indicator_ts].mean().reset_index()
                
                # Erstelle Liniendiagramm
                fig = px.line(
                    df_ts_agg,
                    x='syear',
                    y=selected_indicator_ts,
                    color='matched_bundesland',
                    markers=True,
                    title=f'{selected_indicator_ts} Zeitreihe (2015-2019)'
                )
                
                # Zeige Plot
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

def show_model_performance(model_comparison, df):
    """Zeige Model Performance Page"""
    
    # Titel
    st.header('📈 Model Performance')
    st.markdown('Vergleich aller trainierten ML-Modelle')
    
    # Prüfe ob model_comparison vorhanden
    if model_comparison is None:
        st.warning('Model-Vergleichsdaten nicht gefunden. Bitte Notebook 03 ausführen.')
        return
    
    # Tabs
    tab1, tab2 = st.tabs(['📊 Metriken-Vergleich', '📈 Visualisierungen'])
    
    # ========== TAB 1: METRIKEN ==========
    with tab1:
        st.subheader('📊 Model-Vergleichstabelle')
        
        # Zeige Tabelle
        st.dataframe(model_comparison, use_container_width=True)
        
        # Trennlinie
        st.markdown('---')
        
        # Bestes Modell highlighten
        best_model = model_comparison.iloc[0]
        
        st.success(f'''
        **🏆 Bestes Modell: {best_model["Model"]}**
        
        - **ROC-AUC:** {best_model["ROC-AUC"]:.3f}
        - **Accuracy:** {best_model["Accuracy"]:.3f}
        - **F1-Score:** {best_model["F1-Score"]:.3f}
        ''')
    
    # ========== TAB 2: VISUALISIERUNGEN ==========
    with tab2:
        # Metriken-Vergleich als Balkendiagramm
        st.subheader('📈 Metriken-Vergleich')
        
        # Wähle Metrik
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        selected_metric = st.selectbox('Wähle Metrik', metrics)
        
        # Erstelle Balkendiagramm
        fig = px.bar(
            model_comparison,
            x='Model',
            y=selected_metric,
            title=f'{selected_metric} pro Modell',
            color=selected_metric,
            color_continuous_scale='Blues'
        )
        
        # Zeige Plot
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5: PREDICTION TOOL
# ============================================================

def show_prediction_tool(model, scaler, feature_names, df):
    """Zeige Prediction Tool Page"""
    
    # Titel
    st.header('🎯 Vorhersage-Tool')
    st.markdown('Sage Bildungserfolg für eine neue Person vorher')
    
    # Prüfe ob Modell vorhanden
    if model is None or scaler is None or feature_names is None:
        st.warning('Modell nicht gefunden. Bitte Notebook 03 ausführen.')
        return
    
    # Info-Box
    st.info('''
    **So funktioniert's:**
    1. Gib die Werte für alle Features ein
    2. Klicke auf "Vorhersage berechnen"
    3. Sieh die Wahrscheinlichkeit für High Education (>=12 Jahre)
    ''')
    
    # Trennlinie
    st.markdown('---')
    
    # Input-Form
    st.subheader('📝 Input-Werte')
    
    # Erstelle 3 Spalten für Inputs
    col1, col2, col3 = st.columns(3)
    
    # Dictionary für Input-Werte
    input_values = {}
    
    # Iteriere durch Features
    for idx, feature in enumerate(feature_names):
        # Bestimme Spalte (zyklisch)
        col = [col1, col2, col3][idx % 3]
        
        with col:
            # Bestimme Default-Wert (Median aus Daten)
            if feature in df.columns:
                default_val = float(df[feature].median())
            else:
                default_val = 0.0
            
            # Number Input
            input_values[feature] = st.number_input(
                feature,
                value=default_val,
                key=f'input_{feature}'
            )
    
    # Trennlinie
    st.markdown('---')
    
    # Predict-Button
    if st.button('🎯 Vorhersage berechnen', type='primary', use_container_width=True):
        # Erstelle Feature-Array
        X_input = np.array([list(input_values.values())])
        
        # Standardisiere
        X_scaled = scaler.transform(X_input)
        
        # Vorhersage
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Zeige Ergebnis
        st.markdown('---')
        st.subheader('📊 Ergebnis')
        
        # Spalten für Ausgabe
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            # Vorhersage
            if prediction == 1:
                st.success('✅ **High Education** (>=12 Jahre)')
            else:
                st.error('❌ **Lower Education** (<12 Jahre)')
        
        with res_col2:
            # Wahrscheinlichkeit
            prob_high = probability[1] * 100
            st.metric('Wahrscheinlichkeit High Education', f'{prob_high:.1f}%')
        
        # Progress Bar
        st.progress(prob_high / 100)
        
        # Interpretation
        st.markdown('---')
        st.subheader('💡 Interpretation')
        
        if prob_high >= 70:
            st.success('Hohe Wahrscheinlichkeit für Bildungserfolg!')
        elif prob_high >= 50:
            st.warning('Mittlere Wahrscheinlichkeit. Regionale Förderung könnte helfen.')
        else:
            st.error('Niedrige Wahrscheinlichkeit. Gezielte Interventionen empfohlen.')
