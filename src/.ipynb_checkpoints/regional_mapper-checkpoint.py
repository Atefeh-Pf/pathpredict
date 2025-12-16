"""
Regional Clustering Module
===========================
Creates synthetic regional clusters based on socioeconomic characteristics
Erstellt synthetische regionale Cluster basierend auf sozioökonomischen Merkmalen
"""

# Pandas importieren für Datenmanipulation (DataFrames)
import pandas as pd

# NumPy importieren für numerische Berechnungen (Arrays, Mathe)
import numpy as np

# KMeans importieren für Clustering-Algorithmus
from sklearn.cluster import KMeans

# StandardScaler importieren für Standardisierung der Daten (Mean=0, Std=1)
from sklearn.preprocessing import StandardScaler

# PCA importieren für Dimensionsreduktion (6D -> 2D für Visualisierung)
from sklearn.decomposition import PCA

# Matplotlib importieren für Basis-Plots
import matplotlib.pyplot as plt

# Seaborn importieren für erweiterte Visualisierungen (Heatmaps)
import seaborn as sns


# Klasse definieren die regionale Cluster erstellt
class RegionalClusterMapper:
    """Create synthetic regional clusters from individual-level data"""
    
    # Konstruktor: wird aufgerufen wenn Objekt erstellt wird
    def __init__(self, n_clusters=16, random_state=42):
        """
        Initialize cluster mapper
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create (default: 16, similar to German states)
        random_state : int
            Random seed for reproducibility
        """
        # Speichere Anzahl der Cluster (16 = wie Bundesländer)
        self.n_clusters = n_clusters
        
        # Speichere Random Seed für Reproduzierbarkeit
        self.random_state = random_state
        
        # Erstelle StandardScaler Objekt für spätere Standardisierung
        self.scaler = StandardScaler()
        
        # Initialisiere KMeans als None (wird später erstellt)
        self.kmeans = None
        
        # Initialisiere Cluster-Profile als None (wird später erstellt)
        self.cluster_profiles = None
    
    # Methode zum Erstellen der Cluster
    def create_clusters(self, df, features):
        """
        Create regional clusters based on socioeconomic features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with individual-level data
        features : list
            List of feature column names to use for clustering
            
        Returns:
        --------
        pd.DataFrame
            Original dataframe with added 'region_cluster' column
        """
        # Gib Info aus: wie viele Cluster werden erstellt
        print(f"\nCreating {self.n_clusters} synthetic regional clusters...")
        
        # Gib Info aus: welche Features werden verwendet
        print(f"Clustering features: {features}")
        
        # Kopiere nur die relevanten Spalten aus DataFrame
        X = df[features].copy()
        
        # Fülle fehlende Werte mit Median (robuster als Mean)
        X = X.fillna(X.median())
        
        # Standardisiere Features (Mean=0, Std=1) und speichere Scaler-Parameter
        X_scaled = self.scaler.fit_transform(X)
        
        # Erstelle KMeans Objekt mit n_clusters und random_state
        # n_init=10: Führe 10 Mal mit verschiedenen Startwerten aus, nimm bestes Ergebnis
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                             random_state=self.random_state,
                             n_init=10)
        
        # Führe Clustering durch und speichere Cluster-IDs als neue Spalte
        # fit_predict: trainiert und gibt Vorhersagen zurück
        df['region_cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Gib Erfolgsmeldung aus
        print(f"Created {self.n_clusters} clusters")
        
        # Gib Verteilung der Cluster aus
        print(f"\nCluster distribution:")
        
        # Zähle wie viele Personen in jedem Cluster, sortiere nach Cluster-ID
        print(df['region_cluster'].value_counts().sort_index())
        
        # Gib DataFrame mit neuer region_cluster Spalte zurück
        return df
    
    # Methode zum Erstellen von Cluster-Profilen
    def profile_clusters(self, df, profile_vars):
        """
        Create profiles for each cluster
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with region_cluster column
        profile_vars : list
            Variables to use for profiling
            
        Returns:
        --------
        pd.DataFrame
            Cluster profiles (mean values for each variable)
        """
        # Gib Info aus
        print("\nProfiling clusters...")
        
        # Gruppiere nach Cluster, berechne Durchschnitt für jede Variable
        self.cluster_profiles = df.groupby('region_cluster')[profile_vars].mean()
        
        # Füge Cluster-Größe hinzu (Anzahl Personen pro Cluster)
        self.cluster_profiles['cluster_size'] = df.groupby('region_cluster').size()
        
        # Erstelle beschreibende Labels für jeden Cluster
        self.cluster_profiles = self._create_cluster_labels(self.cluster_profiles)
        
        # Gib Erfolgsmeldung aus
        print("Cluster profiles created")
        
        # Gib Profile zurück
        return self.cluster_profiles
    
    # Private Methode zum Erstellen von Cluster-Labels (Unterstrich = privat)
    def _create_cluster_labels(self, profiles):
        """Create descriptive labels for clusters based on characteristics"""
        
        # Prüfe ob Einkommen und Bildung in Profilen vorhanden
        if 'einkommenj1' in profiles.columns and 'bildung' in profiles.columns:
            
            # Berechne Median von Einkommen
            income_median = profiles['einkommenj1'].median()
            
            # Berechne Median von Bildung
            edu_median = profiles['bildung'].median()
            
            # Leere Liste für Labels
            labels = []
            
            # Iteriere durch jeden Cluster
            for idx, row in profiles.iterrows():
                
                # Bestimme Einkommens-Level (High/Low)
                income_level = "High" if row['einkommenj1'] > income_median else "Low"
                
                # Bestimme Bildungs-Level (High/Low)
                edu_level = "High" if row['bildung'] > edu_median else "Low"
                
                # Erstelle kombiniertes Label und füge zur Liste hinzu
                labels.append(f"{income_level}Inc_{edu_level}Edu")
            
            # Füge Labels als neue Spalte hinzu
            profiles['cluster_label'] = labels
        
        # Gib Profile mit Labels zurück
        return profiles
    
    # Methode zum Visualisieren der Cluster mit PCA
    def visualize_clusters(self, df, features, save_path=None):
        """
        Visualize clusters using PCA
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with region_cluster column
        features : list
            Features used for clustering
        save_path : str, optional
            Path to save visualization
        """
        # Fülle fehlende Werte mit Median
        X = df[features].fillna(df[features].median())
        
        # Transformiere mit bereits trainiertem Scaler (nur transform, nicht fit)
        X_scaled = self.scaler.transform(X)
        
        # Erstelle PCA Objekt für 2D Reduktion
        pca = PCA(n_components=2, random_state=self.random_state)
        
        # Führe PCA durch: 6D -> 2D
        X_pca = pca.fit_transform(X_scaled)
        
        # Erstelle Figure mit 12x8 Größe
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Erstelle Scatter-Plot: X=PC1, Y=PC2, Farbe=Cluster
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],  # X[:,0]=erste Spalte, X[:,1]=zweite Spalte
                            c=df['region_cluster'],      # Farbe basierend auf Cluster
                            cmap='tab20',                # Farbpalette mit 20 Farben
                            alpha=0.6,                   # 60% Transparenz
                            s=50)                        # Punkt-Größe
        
        # Setze X-Achsen-Label mit Varianz-Anteil
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        
        # Setze Y-Achsen-Label mit Varianz-Anteil
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Setze Titel
        ax.set_title('Synthetic Regional Clusters (PCA Visualization)')
        
        # Füge Colorbar hinzu (Legende für Cluster-Farben)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        # Wenn Speicherpfad angegeben, speichere als PNG
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Optimiere Layout
        plt.tight_layout()
        
        # Gib Figure zurück
        return fig
    
    # Methode zum Visualisieren der Cluster-Profile als Heatmap
    def visualize_cluster_profiles(self, save_path=None):
        """
        Create heatmap of cluster profiles
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save visualization
        """
        # Prüfe ob Profile existieren
        if self.cluster_profiles is None:
            raise ValueError("Cluster profiles not created. Call profile_clusters() first.")
        
        # Wähle nur numerische Spalten
        numeric_cols = self.cluster_profiles.select_dtypes(include=[np.number]).columns
        
        # Entferne cluster_size aus Liste (soll nicht in Heatmap)
        numeric_cols = [c for c in numeric_cols if c != 'cluster_size']
        
        # Normalisiere Werte (z-Score: (x-mean)/std) für bessere Visualisierung
        profiles_norm = self.cluster_profiles[numeric_cols].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        
        # Erstelle Figure mit 12x10 Größe
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Erstelle Heatmap mit Seaborn
        sns.heatmap(profiles_norm.T,          # Transponiert (Features als Zeilen)
                   cmap='RdYlBu_r',            # Farbpalette (Rot-Gelb-Blau, umgekehrt)
                   center=0,                   # Zentriere Farbskala bei 0
                   annot=True,                 # Zeige Werte in Zellen
                   fmt='.2f',                  # Format: 2 Dezimalstellen
                   cbar_kws={'label': 'Standardized Value'},  # Colorbar-Label
                   ax=ax)                      # Zeichne in ax
        
        # Setze X-Achsen-Label
        ax.set_xlabel('Cluster ID')
        
        # Setze Y-Achsen-Label
        ax.set_ylabel('Feature')
        
        # Setze Titel
        ax.set_title('Cluster Profiles (Standardized Values)')
        
        # Wenn Speicherpfad angegeben, speichere als PNG
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved profile heatmap to {save_path}")
        
        # Optimiere Layout
        plt.tight_layout()
        
        # Gib Figure zurück
        return fig


# Funktion zum Erstellen der regionalen Datenzuordnung (außerhalb der Klasse)
def create_regional_data_mapping():
    """
    Create mapping of synthetic clusters to real regional average data
    
    This function creates a DataFrame with average regional characteristics
    based on real German data from INKAR (www.inkar.de), 2015-2019.
    
    Data Source: INKAR - Indikatoren und Karten zur Raum- und Stadtentwicklung
    Bundesinstitut für Bau-, Stadt- und Raumforschung (BBSR)
    Lizenz: Datenlizenz Deutschland – Namensnennung – Version 2.0
    
    Returns:
    --------
    pd.DataFrame
        Regional characteristics for each cluster (mapped to 16 Bundesländer)
    """
    
    # ECHTE Daten aus INKAR 2015-2019 - 16 Bundesländer
    # Reihenfolge: SH, HH, NI, HB, NW, HE, RP, BW, BY, SL, BE, BB, MV, SN, ST, TH
    
    # Erstelle DataFrame mit echten Regionaldaten
    regional_data = pd.DataFrame({
        
        # Cluster-IDs von 0 bis 15 (range(16) = [0,1,2,...,15])
        'region_cluster': range(16),
        
        # Bundesländer-Namen als Liste
        'bundesland': [
            'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
            'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg',
            'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
            'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen'
        ],
        
        # Arbeitslosenquote 2019 in Prozent (Quelle: INKAR)
        'unemployment_rate': [5.07, 6.13, 5.04, 9.93, 6.55, 4.40, 4.35, 3.16,
                              2.84, 6.16, 7.82, 5.77, 7.12, 5.46, 7.15, 5.27],
        
        # Schulabgänger ohne Abschluss 2019 in Prozent (Quelle: INKAR)
        'school_dropout_rate': [9.18, 6.02, 6.77, 9.26, 6.03, 5.21, 7.47, 5.90,
                                5.42, 7.27, 8.94, 7.49, 9.25, 8.65, 11.26, 8.94],
        
        # Abiturquote 2019 in Prozent - Anteil mit allgemeiner Hochschulreife (Quelle: INKAR)
        'abitur_rate': [36.09, 53.87, 33.78, 35.93, 39.40, 32.32, 37.19, 29.94,
                        28.71, 34.44, 44.44, 40.17, 35.53, 33.20, 29.65, 32.73],
        
        # Kinderarmut 2019 in Prozent - Anteil Kinder in Haushalten mit Grundsicherung (Quelle: INKAR)
        'child_poverty_rate': [14.91, 18.98, 13.94, 30.88, 18.17, 13.68, 11.21, 7.98,
                               6.14, 18.90, 26.31, 11.88, 14.39, 11.53, 17.35, 12.00],
        
        # Betreuungsquote Vorschulkinder 2019 in Prozent (Quelle: INKAR)
        'childcare_rate': [88.99, 86.56, 89.20, 84.10, 88.31, 88.97, 91.42, 91.27,
                           89.46, 89.67, 89.34, 91.28, 94.00, 93.33, 92.43, 95.08],
        
        # Medianeinkommen 2019 in Euro pro Monat (Quelle: INKAR)
        'median_income': [3149.20, 3820.41, 3335.40, 3596.06, 3509.19, 3788.91, 3433.55, 3824.39,
                          3676.05, 3474.86, 3383.33, 2728.84, 2630.54, 2745.94, 2720.05, 2689.95]
    })
    
    # Erstelle Region-Type Labels basierend auf Risiko-Faktoren
    # conditions = Liste von Bedingungen
    conditions = [
        # High_Risk: Arbeitslosigkeit > 7% UND Kinderarmut > 20%
        (regional_data['unemployment_rate'] > 7) & (regional_data['child_poverty_rate'] > 20),
        
        # Medium_Risk: Arbeitslosigkeit > 6% UND Kinderarmut > 15%
        (regional_data['unemployment_rate'] > 6) & (regional_data['child_poverty_rate'] > 15),
        
        # Low_Risk: Arbeitslosigkeit <= 4% UND Kinderarmut < 10%
        (regional_data['unemployment_rate'] <= 4) & (regional_data['child_poverty_rate'] < 10),
    ]
    
    # Labels für jede Bedingung
    labels = ['High_Risk', 'Medium_Risk', 'Low_Risk']
    
    # np.select: Wählt Label basierend auf erster zutreffender Bedingung
    # default='Medium_Low_Risk': Wenn keine Bedingung zutrifft
    regional_data['region_type'] = np.select(conditions, labels, default='Medium_Low_Risk')
    
    # Gib fertigen DataFrame zurück
    return regional_data


# Wird nur ausgeführt wenn Datei direkt gestartet wird (nicht bei Import)
if __name__ == "__main__":
    
    # Gib Modul-Info aus
    print("Regional Clustering Module")
    
    # Gib Beschreibung aus
    print("\nThis module creates synthetic regional clusters and maps them to real data.")