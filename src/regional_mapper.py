"""
Regional Clustering Module
===========================
Erstellt synthetische regionale Cluster und verknüpft sie mit INKAR-Daten (2015-2019)

Datenquelle: INKAR - Bundesinstitut für Bau-, Stadt- und Raumforschung (BBSR)
Lizenz: Datenlizenz Deutschland – Namensnennung – Version 2.0
"""

# Pandas importieren für Datenmanipulation (DataFrames)
import pandas as pd

# NumPy importieren für numerische Berechnungen (Arrays, Mathe)
import numpy as np

# KMeans importieren für Clustering-Algorithmus
from sklearn.cluster import KMeans

# StandardScaler importieren für Standardisierung der Daten
from sklearn.preprocessing import StandardScaler

# PCA importieren für Dimensionsreduktion
from sklearn.decomposition import PCA

# Matplotlib importieren für Plots
import matplotlib.pyplot as plt

# Seaborn importieren für Heatmaps
import seaborn as sns

# cdist importieren für Distanz-Berechnung
from scipy.spatial.distance import cdist

# linear_sum_assignment importieren für Hungarian Algorithm
from scipy.optimize import linear_sum_assignment


# ============================================================
# INKAR DATEN - ALLE JAHRE 2015-2019
# ============================================================

# INKAR-Rohdaten als Dictionary (Wide-Format)
# Quelle: INKAR - Laufende Raumbeobachtung des BBSR
INKAR_DATA = {
    'bundesland': [
        'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
        'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg',
        'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
        'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen'
    ],
    # Arbeitslosenquote (%)
    'arbeitslosenquote_2015': [6.78, 7.55, 6.11, 10.88, 8.02, 5.26, 5.21, 3.81, 3.62, 7.28, 10.67, 8.68, 10.41, 8.19, 10.24, 7.39],
    'arbeitslosenquote_2016': [6.50, 7.45, 5.97, 10.46, 7.70, 5.13, 5.07, 3.75, 3.47, 7.16, 9.78, 7.99, 9.76, 7.46, 9.88, 6.88],
    'arbeitslosenquote_2017': [6.27, 7.08, 5.72, 10.17, 7.37, 4.87, 4.80, 3.50, 3.18, 6.72, 9.01, 7.11, 8.96, 6.73, 9.14, 6.13],
    'arbeitslosenquote_2018': [5.48, 6.32, 5.14, 10.00, 6.80, 4.46, 4.38, 3.18, 2.87, 6.14, 8.05, 6.14, 7.65, 5.88, 7.67, 5.47],
    'arbeitslosenquote_2019': [5.07, 6.13, 5.04, 9.93, 6.55, 4.40, 4.35, 3.16, 2.84, 6.16, 7.82, 5.77, 7.12, 5.46, 7.15, 5.27],
    # Schulabgänger ohne Abschluss (%)
    'schulabg_ohne_abschluss_2015': [7.52, 5.47, 5.86, 8.06, 5.47, 4.88, 6.34, 5.48, 5.02, 6.95, 9.32, 6.79, 9.00, 8.25, 11.43, 8.08],
    'schulabg_ohne_abschluss_2016': [6.62, 5.85, 5.78, 8.09, 5.46, 4.86, 6.65, 5.50, 5.11, 6.19, 8.52, 6.80, 8.76, 8.09, 10.28, 8.25],
    'schulabg_ohne_abschluss_2017': [7.65, 5.29, 5.97, 8.17, 5.87, 4.91, 6.70, 5.57, 5.26, 6.89, 8.60, 7.42, 8.73, 8.38, 9.93, 8.42],
    'schulabg_ohne_abschluss_2018': [8.49, 6.39, 6.04, 8.49, 5.95, 4.94, 7.22, 5.76, 5.32, 6.75, 8.60, 7.33, 8.81, 8.42, 10.30, 8.76],
    'schulabg_ohne_abschluss_2019': [9.18, 6.02, 6.77, 9.26, 6.03, 5.21, 7.47, 5.90, 5.42, 7.27, 8.94, 7.49, 9.25, 8.65, 11.26, 8.94],
    # Abiturquote (%)
    'abiturquote_2015': [33.59, 55.14, 32.51, 33.81, 37.86, 32.34, 32.69, 27.30, 27.07, 31.22, 44.20, 34.02, 29.60, 29.86, 27.15, 30.56],
    'abiturquote_2016': [45.97, 55.59, 37.97, 36.37, 40.48, 35.54, 35.61, 44.31, 40.66, 37.70, 45.19, 39.38, 33.03, 36.18, 28.07, 36.38],
    'abiturquote_2017': [36.99, 52.84, 34.06, 36.25, 39.35, 33.78, 36.47, 29.71, 28.80, 33.51, 44.93, 38.22, 34.63, 32.11, 29.03, 32.62],
    'abiturquote_2018': [35.38, 52.75, 33.23, 35.37, 38.84, 31.75, 36.90, 29.68, 28.48, 34.39, 44.14, 39.15, 34.85, 32.43, 29.47, 32.28],
    'abiturquote_2019': [36.09, 53.87, 33.78, 35.93, 39.40, 32.32, 37.19, 29.94, 28.71, 34.44, 44.44, 40.17, 35.53, 33.20, 29.65, 32.73],
    # Kinderarmut (%) - nur ab 2016
    'kinderarmut_2016': [15.51, 19.47, 14.47, 31.24, 18.50, 14.19, 11.82, 8.44, 6.54, 18.11, 28.20, 13.36, 16.33, 13.05, 19.19, 13.68],
    'kinderarmut_2017': [16.06, 19.98, 14.72, 31.74, 18.87, 14.39, 11.64, 8.45, 6.94, 19.26, 27.79, 12.74, 15.56, 12.50, 18.66, 13.00],
    'kinderarmut_2018': [15.60, 19.65, 14.51, 31.24, 18.66, 14.16, 11.47, 8.22, 6.62, 19.32, 27.31, 12.35, 15.03, 12.04, 18.22, 12.58],
    'kinderarmut_2019': [14.91, 18.98, 13.94, 30.88, 18.17, 13.68, 11.21, 7.98, 6.14, 18.90, 26.31, 11.88, 14.39, 11.53, 17.35, 12.00],
    # Betreuungsquote Vorschulkinder (%)
    'betreuungsquote_2015': [91.00, 89.60, 90.40, 86.40, 87.40, 90.10, 93.60, 93.40, 90.20, 91.50, 92.40, 93.80, 94.10, 95.00, 93.50, 95.90],
    'betreuungsquote_2016': [91.17, 86.90, 89.65, 84.33, 88.18, 89.88, 92.58, 92.95, 90.12, 90.95, 91.27, 93.09, 94.74, 95.35, 93.75, 96.17],
    'betreuungsquote_2017': [89.21, 85.87, 89.11, 83.90, 88.00, 89.47, 91.83, 92.08, 89.79, 90.29, 90.02, 92.17, 93.80, 94.36, 93.16, 95.44],
    'betreuungsquote_2018': [89.37, 87.07, 89.35, 84.00, 88.39, 89.30, 91.42, 91.62, 89.51, 89.82, 89.73, 91.49, 94.25, 93.58, 92.67, 95.39],
    'betreuungsquote_2019': [88.99, 86.56, 89.20, 84.10, 88.31, 88.97, 91.42, 91.27, 89.46, 89.67, 89.34, 91.28, 94.00, 93.33, 92.43, 95.08],
    # Medianeinkommen (Euro/Monat)
    'medianeinkommen_2015': [2856.08, 3487.03, 3006.12, 3241.01, 3173.89, 3431.19, 3085.75, 3465.01, 3323.93, 3101.45, 3057.34, 2421.78, 2316.70, 2422.69, 2392.18, 2375.90],
    'medianeinkommen_2016': [2902.39, 3543.80, 3067.57, 3296.00, 3233.24, 3496.22, 3162.60, 3538.03, 3375.07, 3181.13, 3117.76, 2476.04, 2377.58, 2479.72, 2452.66, 2425.70],
    'medianeinkommen_2017': [2971.65, 3618.80, 3155.31, 3387.52, 3318.51, 3586.68, 3250.67, 3628.13, 3459.81, 3278.27, 3196.56, 2551.52, 2455.24, 2561.61, 2530.48, 2504.35],
    'medianeinkommen_2018': [3062.37, 3717.62, 3250.87, 3499.22, 3417.48, 3694.30, 3345.94, 3732.89, 3571.39, 3379.93, 3297.81, 2643.94, 2546.28, 2657.79, 2625.11, 2597.44],
    'medianeinkommen_2019': [3149.20, 3820.41, 3335.40, 3596.06, 3509.19, 3788.91, 3433.55, 3824.39, 3676.05, 3474.86, 3383.33, 2728.84, 2630.54, 2745.94, 2720.05, 2689.95]
}


def create_inkar_long_format():
    """
    Erstellt INKAR-Daten im Long-Format (eine Zeile pro Bundesland pro Jahr)
    
    Returns:
    --------
    pd.DataFrame mit Spalten:
        - bundesland: Name des Bundeslandes
        - year: Jahr (2015-2019)
        - arbeitslosenquote: Arbeitslosenquote (%)
        - schulabg_ohne_abschluss: Schulabgänger ohne Abschluss (%)
        - abiturquote: Abiturquote (%)
        - kinderarmut: Kinderarmut (%) - NaN für 2015
        - betreuungsquote: Betreuungsquote Vorschulkinder (%)
        - medianeinkommen: Medianeinkommen (Euro/Monat)
    """
    # Leere Liste für Long-Format Einträge
    long_data = []
    
    # Iteriere durch alle 16 Bundesländer
    for i, bundesland in enumerate(INKAR_DATA['bundesland']):
        
        # Iteriere durch Jahre 2015-2019
        for year in [2015, 2016, 2017, 2018, 2019]:
            
            # Erstelle Eintrag für dieses Bundesland und Jahr
            entry = {
                'bundesland': bundesland,
                'year': year,
                'arbeitslosenquote': INKAR_DATA[f'arbeitslosenquote_{year}'][i],
                'schulabg_ohne_abschluss': INKAR_DATA[f'schulabg_ohne_abschluss_{year}'][i],
                'abiturquote': INKAR_DATA[f'abiturquote_{year}'][i],
                'betreuungsquote': INKAR_DATA[f'betreuungsquote_{year}'][i],
                'medianeinkommen': INKAR_DATA[f'medianeinkommen_{year}'][i],
            }
            
            # Kinderarmut nur ab 2016 verfügbar
            if year >= 2016:
                # Hole Kinderarmut-Wert für dieses Jahr
                entry['kinderarmut'] = INKAR_DATA[f'kinderarmut_{year}'][i]
            else:
                # 2015: setze NaN
                entry['kinderarmut'] = np.nan
            
            # Füge Eintrag zur Liste hinzu
            long_data.append(entry)
    
    # Erstelle DataFrame aus Liste
    df = pd.DataFrame(long_data)
    
    # Gib DataFrame zurück
    return df


def create_inkar_averaged():
    """
    Erstellt INKAR-Daten als Durchschnitt über alle Jahre (für Matching)
    
    Returns:
    --------
    pd.DataFrame mit Durchschnittswerten 2015-2019 pro Bundesland
    """
    # Hole Long-Format Daten
    df_long = create_inkar_long_format()
    
    # Gruppiere nach Bundesland und berechne Durchschnitt
    df_avg = df_long.groupby('bundesland').agg({
        'arbeitslosenquote': 'mean',
        'schulabg_ohne_abschluss': 'mean',
        'abiturquote': 'mean',
        'kinderarmut': 'mean',  # Durchschnitt 2016-2019 (2015 ist NaN)
        'betreuungsquote': 'mean',
        'medianeinkommen': 'mean'
    }).reset_index()
    
    # Gib DataFrame zurück
    return df_avg


# ============================================================
# CLUSTERING KLASSE
# ============================================================

class RegionalClusterMapper:
    """Erstellt synthetische regionale Cluster aus Individualdaten"""
    
    def __init__(self, n_clusters=16, random_state=42):
        """
        Initialisiere Cluster Mapper
        
        Parameters:
        -----------
        n_clusters : int
            Anzahl Cluster (16 = wie Bundesländer)
        random_state : int
            Seed für Reproduzierbarkeit
        """
        # Speichere Anzahl Cluster
        self.n_clusters = n_clusters
        
        # Speichere Random Seed
        self.random_state = random_state
        
        # Erstelle Scaler für Standardisierung
        self.scaler = StandardScaler()
        
        # Initialisiere KMeans als None
        self.kmeans = None
        
        # Initialisiere Cluster-Profile als None
        self.cluster_profiles = None
        
    def create_clusters(self, df, features):
        """
        Erstelle Cluster basierend auf Features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        features : list
            Liste der Feature-Spalten
            
        Returns:
        --------
        pd.DataFrame mit neuer 'region_cluster' Spalte
        """
        # Info ausgeben
        print(f"\nErstelle {self.n_clusters} synthetische Cluster...")
        print(f"Features: {features}")
        
        # Kopiere Features
        X = df[features].copy()
        
        # Fülle NaN mit Median
        X = X.fillna(X.median())
        
        # Standardisiere
        X_scaled = self.scaler.fit_transform(X)
        
        # Erstelle KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        # Führe Clustering durch
        df['region_cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Info ausgeben
        print(f"Erstellt: {self.n_clusters} Cluster")
        print(f"\nCluster-Verteilung:")
        print(df['region_cluster'].value_counts().sort_index())
        
        # Gib DataFrame zurück
        return df
    
    def profile_clusters(self, df, profile_vars):
        """Erstelle Profile für jeden Cluster"""
        
        # Info ausgeben
        print("\nErstelle Cluster-Profile...")
        
        # Berechne Durchschnitt pro Cluster
        self.cluster_profiles = df.groupby('region_cluster')[profile_vars].mean()
        
        # Füge Cluster-Größe hinzu
        self.cluster_profiles['cluster_size'] = df.groupby('region_cluster').size()
        
        # Info ausgeben
        print("Cluster-Profile erstellt")
        
        # Gib Profile zurück
        return self.cluster_profiles
    
    def visualize_cluster_profiles(self, save_path=None):
        """Erstelle Heatmap der Cluster-Profile"""
        
        # Prüfe ob Profile existieren
        if self.cluster_profiles is None:
            raise ValueError("Keine Profile. Zuerst profile_clusters() aufrufen.")
        
        # Wähle numerische Spalten
        numeric_cols = self.cluster_profiles.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'cluster_size']
        
        # Normalisiere
        profiles_norm = self.cluster_profiles[numeric_cols].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        
        # Erstelle Figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Zeichne Heatmap
        sns.heatmap(
            profiles_norm.T,
            cmap='RdYlBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Standardisierter Wert'},
            ax=ax
        )
        
        # Setze Labels
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Feature')
        ax.set_title('Cluster-Profile (standardisiert)')
        
        # Speichere wenn Pfad angegeben
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gespeichert: {save_path}")
        
        # Optimiere Layout
        plt.tight_layout()
        
        # Gib Figure zurück
        return fig


# ============================================================
# MATCHING FUNKTIONEN
# ============================================================

def match_clusters_to_bundeslaender(cluster_profiles, 
                                     cluster_features=['einkommenj1', 'bildung'],
                                     inkar_features=['medianeinkommen', 'abiturquote']):
    """
    Matche Cluster zu Bundesländern mit Hungarian Algorithm
    
    Verwendet Durchschnittswerte 2015-2019 für stabiles Matching.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        Durchschnittswerte pro Cluster
    cluster_features : list
        SOEP-Features für Vergleich
    inkar_features : list
        INKAR-Features für Vergleich
    
    Returns:
    --------
    dict mit:
        - mapping: cluster_id -> bundesland
        - results: DataFrame mit Details
        - quality: Qualitäts-Statistiken
    """
    # Hole INKAR-Durchschnittswerte
    inkar_avg = create_inkar_averaged()
    
    # Prüfe Features
    for f in cluster_features:
        if f not in cluster_profiles.columns:
            raise ValueError(f"Feature '{f}' nicht in cluster_profiles")
    for f in inkar_features:
        if f not in inkar_avg.columns:
            raise ValueError(f"Feature '{f}' nicht in inkar_avg")
    
    # Extrahiere und normalisiere SOEP-Daten
    soep_data = cluster_profiles[cluster_features].copy()
    soep_norm = (soep_data - soep_data.mean()) / soep_data.std()
    
    # Extrahiere und normalisiere INKAR-Daten
    inkar_data = inkar_avg[inkar_features].copy()
    inkar_norm = (inkar_data - inkar_data.mean()) / inkar_data.std()
    
    # Berechne Distanz-Matrix
    distances = cdist(soep_norm.values, inkar_norm.values, metric='euclidean')
    
    # Hungarian Algorithm
    cluster_indices, bundesland_indices = linear_sum_assignment(distances)
    
    # Erstelle Ergebnisse
    results = []
    mapping = {}
    
    # Iteriere durch Zuordnungen
    for cluster_id, bl_idx in zip(cluster_indices, bundesland_indices):
        # Hole Bundesland-Name
        bl_name = inkar_avg.iloc[bl_idx]['bundesland']
        
        # Hole Distanz
        dist = distances[cluster_id, bl_idx]
        
        # Speichere Mapping
        mapping[int(cluster_id)] = bl_name
        
        # Füge zu Ergebnissen hinzu
        results.append({
            'cluster_id': int(cluster_id),
            'bundesland': bl_name,
            'distanz': round(dist, 3),
            'quality': calculate_match_quality(dist)
        })
    
    # Erstelle DataFrame
    results_df = pd.DataFrame(results).sort_values('cluster_id')
    
    # Berechne Qualitäts-Statistiken
    quality_stats = calculate_quality_statistics(results_df)
    
    # Gib Ergebnisse zurück
    return {
        'mapping': mapping,
        'results': results_df,
        'quality': quality_stats
    }


def calculate_match_quality(distance):
    """
    Berechne Match-Qualität basierend auf Distanz
    
    Schwellenwerte:
    - SEHR_GUT: < 0.5
    - GUT: 0.5 - 1.0
    - MITTEL: 1.0 - 1.5
    - SCHWACH: > 1.5
    """
    # Prüfe Schwellenwerte
    if distance < 0.5:
        return 'SEHR_GUT'
    elif distance < 1.0:
        return 'GUT'
    elif distance < 1.5:
        return 'MITTEL'
    else:
        return 'SCHWACH'


def calculate_quality_statistics(results_df):
    """Berechne Qualitäts-Statistiken"""
    
    # Zähle pro Qualitätsstufe
    quality_counts = results_df['quality'].value_counts()
    
    # Alle Kategorien
    all_categories = ['SEHR_GUT', 'GUT', 'MITTEL', 'SCHWACH']
    counts = {cat: quality_counts.get(cat, 0) for cat in all_categories}
    
    # Berechne Prozentsätze
    total = len(results_df)
    percentages = {cat: round(counts[cat] / total * 100, 1) for cat in all_categories}
    
    # Berechne Distanz-Statistiken
    total_distance = results_df['distanz'].sum()
    mean_distance = results_df['distanz'].mean()
    
    # Prüfe ob bestanden (>= 50% GUT oder besser)
    good_matches = counts['SEHR_GUT'] + counts['GUT']
    passed = good_matches >= total / 2
    
    # Gib Statistiken zurück
    return {
        'counts': counts,
        'percentages': percentages,
        'total_distance': round(total_distance, 2),
        'mean_distance': round(mean_distance, 3),
        'good_matches': good_matches,
        'passed': passed
    }


def print_matching_report(matching_results):
    """Drucke formatierten Matching-Report"""
    
    # Hole Daten
    results_df = matching_results['results']
    quality = matching_results['quality']
    
    # Drucke Header
    print("=" * 70)
    print("CLUSTER-BUNDESLAND MATCHING REPORT")
    print("=" * 70)
    
    # Sortiere nach Distanz
    sorted_df = results_df.sort_values('distanz')
    
    # Drucke Zuordnungen
    print("\n--- Zuordnungen (sortiert nach Qualität) ---\n")
    for _, row in sorted_df.iterrows():
        print(f"Cluster {row['cluster_id']:2.0f} -> {row['bundesland']:25s} | "
              f"Distanz: {row['distanz']:.2f} | {row['quality']}")
    
    # Drucke Qualitäts-Übersicht
    print("\n--- Qualitäts-Übersicht ---\n")
    print(f"SEHR_GUT (< 0.5):  {quality['counts']['SEHR_GUT']:2d} ({quality['percentages']['SEHR_GUT']}%)")
    print(f"GUT (0.5 - 1.0):   {quality['counts']['GUT']:2d} ({quality['percentages']['GUT']}%)")
    print(f"MITTEL (1.0 - 1.5):{quality['counts']['MITTEL']:2d} ({quality['percentages']['MITTEL']}%)")
    print(f"SCHWACH (> 1.5):   {quality['counts']['SCHWACH']:2d} ({quality['percentages']['SCHWACH']}%)")
    
    # Drucke Distanz-Statistiken
    print(f"\nGesamtdistanz: {quality['total_distance']}")
    print(f"Durchschnittsdistanz: {quality['mean_distance']}")
    
    # Drucke Gesamtbewertung
    print("\n--- Gesamtbewertung ---\n")
    if quality['passed']:
        print(f"BESTANDEN: {quality['good_matches']}/16 sind GUT oder SEHR_GUT")
    else:
        print(f"NICHT BESTANDEN: Nur {quality['good_matches']}/16 sind GUT oder SEHR_GUT")
    
    print("=" * 70)


def merge_inkar_by_year(df, mapping, year_column='syear'):
    """
    Merge INKAR-Daten Jahr-spezifisch basierend auf Matching
    
    Parameters:
    -----------
    df : pd.DataFrame
        SOEP DataFrame mit 'region_cluster' und 'syear'
    mapping : dict
        cluster_id -> bundesland Mapping
    year_column : str
        Name der Jahr-Spalte
    
    Returns:
    --------
    pd.DataFrame mit INKAR-Spalten hinzugefügt
    """
    # Prüfe ob benötigte Spalten existieren
    if 'region_cluster' not in df.columns:
        raise ValueError("Spalte 'region_cluster' nicht gefunden")
    if year_column not in df.columns:
        raise ValueError(f"Spalte '{year_column}' nicht gefunden")
    
    # Erstelle Mapping-DataFrame
    mapping_df = pd.DataFrame([
        {'region_cluster': k, 'matched_bundesland': v}
        for k, v in mapping.items()
    ])
    
    # Merge Mapping zum DataFrame
    df_merged = df.merge(mapping_df, on='region_cluster', how='left')
    
    # Hole INKAR Long-Format Daten
    inkar_long = create_inkar_long_format()
    
    # Benenne Spalten für Merge um
    inkar_long = inkar_long.rename(columns={
        'bundesland': 'matched_bundesland',
        'year': year_column
    })
    
    # Merge INKAR-Daten basierend auf Bundesland UND Jahr
    df_final = df_merged.merge(
        inkar_long,
        on=['matched_bundesland', year_column],
        how='left'
    )
    
    # Gib DataFrame zurück
    return df_final


# ============================================================
# UNIT TESTS
# ============================================================

def test_inkar_long_format():
    """Test create_inkar_long_format()"""
    
    # Erstelle Long-Format
    df = create_inkar_long_format()
    
    # Prüfe Shape: 16 Bundesländer × 5 Jahre = 80 Zeilen
    assert df.shape[0] == 80, f"Erwartet 80 Zeilen, bekommen {df.shape[0]}"
    
    # Prüfe Spalten
    expected_cols = ['bundesland', 'year', 'arbeitslosenquote', 'schulabg_ohne_abschluss',
                     'abiturquote', 'kinderarmut', 'betreuungsquote', 'medianeinkommen']
    for col in expected_cols:
        assert col in df.columns, f"Spalte '{col}' fehlt"
    
    # Prüfe Jahre
    assert set(df['year'].unique()) == {2015, 2016, 2017, 2018, 2019}, "Jahre nicht korrekt"
    
    # Prüfe Bundesländer
    assert df['bundesland'].nunique() == 16, "Nicht 16 Bundesländer"
    
    # Prüfe NaN für Kinderarmut 2015
    nan_2015 = df[df['year'] == 2015]['kinderarmut'].isna().sum()
    assert nan_2015 == 16, f"Kinderarmut 2015: Erwartet 16 NaN, bekommen {nan_2015}"
    
    # Prüfe keine NaN für Kinderarmut 2016-2019
    nan_other = df[df['year'] > 2015]['kinderarmut'].isna().sum()
    assert nan_other == 0, f"Kinderarmut 2016-2019: Erwartet 0 NaN, bekommen {nan_other}"
    
    print("test_inkar_long_format: PASSED")


def test_calculate_match_quality():
    """Test calculate_match_quality()"""
    
    # Teste Schwellenwerte
    assert calculate_match_quality(0.0) == 'SEHR_GUT'
    assert calculate_match_quality(0.49) == 'SEHR_GUT'
    assert calculate_match_quality(0.5) == 'GUT'
    assert calculate_match_quality(0.99) == 'GUT'
    assert calculate_match_quality(1.0) == 'MITTEL'
    assert calculate_match_quality(1.49) == 'MITTEL'
    assert calculate_match_quality(1.5) == 'SCHWACH'
    assert calculate_match_quality(5.0) == 'SCHWACH'
    
    print("test_calculate_match_quality: PASSED")


def test_merge_by_year():
    """Test merge_inkar_by_year()"""
    
    # Erstelle Test-DataFrame
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'region_cluster': [0, 0, 1, 1, 2],
        'syear': [2015, 2019, 2016, 2018, 2017]
    })
    
    # Erstelle Test-Mapping
    test_mapping = {
        0: 'Bayern',
        1: 'Berlin',
        2: 'Hamburg'
    }
    
    # Führe Merge durch
    result = merge_inkar_by_year(test_df, test_mapping, 'syear')
    
    # Prüfe ob alle Zeilen erhalten
    assert len(result) == 5, f"Erwartet 5 Zeilen, bekommen {len(result)}"
    
    # Prüfe ob matched_bundesland korrekt
    assert result.iloc[0]['matched_bundesland'] == 'Bayern'
    assert result.iloc[2]['matched_bundesland'] == 'Berlin'
    assert result.iloc[4]['matched_bundesland'] == 'Hamburg'
    
    # Prüfe ob INKAR-Spalten hinzugefügt
    assert 'arbeitslosenquote' in result.columns
    assert 'abiturquote' in result.columns
    assert 'kinderarmut' in result.columns
    
    # Prüfe Kinderarmut 2015 = NaN
    row_2015 = result[(result['syear'] == 2015) & (result['matched_bundesland'] == 'Bayern')]
    assert row_2015['kinderarmut'].isna().all(), "Kinderarmut 2015 sollte NaN sein"
    
    # Prüfe Kinderarmut 2019 != NaN
    row_2019 = result[(result['syear'] == 2019) & (result['matched_bundesland'] == 'Bayern')]
    assert not row_2019['kinderarmut'].isna().any(), "Kinderarmut 2019 sollte nicht NaN sein"
    
    # Prüfe Jahr-spezifische Werte (Bayern 2015 vs 2019)
    bayern_2015 = result[(result['syear'] == 2015) & (result['matched_bundesland'] == 'Bayern')]['arbeitslosenquote'].values[0]
    bayern_2019 = result[(result['syear'] == 2019) & (result['matched_bundesland'] == 'Bayern')]['arbeitslosenquote'].values[0]
    assert bayern_2015 != bayern_2019, "Arbeitslosenquote sollte unterschiedlich sein für 2015 vs 2019"
    
    print("test_merge_by_year: PASSED")


def test_all_bundeslaender_present():
    """Test ob alle 16 Bundesländer in INKAR-Daten"""
    
    # Erwartete Bundesländer
    expected = [
        'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
        'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg',
        'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
        'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen'
    ]
    
    # Hole Long-Format
    df = create_inkar_long_format()
    
    # Hole unique Bundesländer
    actual = df['bundesland'].unique().tolist()
    
    # Prüfe
    for bl in expected:
        assert bl in actual, f"Bundesland '{bl}' fehlt"
    
    assert len(actual) == 16, f"Erwartet 16 Bundesländer, bekommen {len(actual)}"
    
    print("test_all_bundeslaender_present: PASSED")


def test_all_years_present():
    """Test ob alle Jahre 2015-2019 in INKAR-Daten"""
    
    # Hole Long-Format
    df = create_inkar_long_format()
    
    # Prüfe Jahre
    for year in [2015, 2016, 2017, 2018, 2019]:
        count = len(df[df['year'] == year])
        assert count == 16, f"Jahr {year}: Erwartet 16 Einträge, bekommen {count}"
    
    print("test_all_years_present: PASSED")


def run_all_tests():
    """Führe alle Unit Tests aus"""
    
    print("\n" + "=" * 50)
    print("RUNNING UNIT TESTS")
    print("=" * 50 + "\n")
    
    test_inkar_long_format()
    test_calculate_match_quality()
    test_all_bundeslaender_present()
    test_all_years_present()
    test_merge_by_year()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Regional Clustering Module")
    print("INKAR-Daten: 2015-2019")
    print("\nRunning tests...")
    run_all_tests()
