"""
Integrationstest für regional_mapper.py
========================================
Testet den vollständigen Workflow: Clustering -> Matching -> Merge

Führe aus mit: python test_integration.py
"""

# Importiere Pandas für DataFrames
import pandas as pd

# Importiere NumPy für numerische Operationen
import numpy as np

# Importiere zu testende Funktionen
from regional_mapper import (
    create_inkar_long_format,
    match_clusters_to_bundeslaender,
    merge_inkar_by_year,
    print_matching_report,
    RegionalClusterMapper
)


def test_inkar_long_format_completeness():
    """
    Test 1: INKAR Long-Format Vollständigkeit
    
    Prüft ob alle 80 Kombinationen (16 BL × 5 Jahre) vorhanden sind.
    """
    # Header
    print("\n--- Test 1: INKAR Long-Format ---")
    
    # Hole Long-Format
    inkar_long = create_inkar_long_format()
    
    # Erwartete Kombinationen
    expected = 16 * 5
    actual = len(inkar_long)
    
    # Ausgabe
    print(f"Kombinationen: {actual} (erwartet: {expected})")
    
    # Prüfe
    assert actual == expected, f"FEHLER: {actual} != {expected}"
    
    # Prüfe jede Kombination
    bundeslaender = inkar_long['bundesland'].unique()
    years = [2015, 2016, 2017, 2018, 2019]
    
    # Iteriere durch alle Kombinationen
    for bl in bundeslaender:
        for year in years:
            # Zähle Einträge
            count = len(inkar_long[(inkar_long['bundesland'] == bl) & (inkar_long['year'] == year)])
            # Prüfe
            assert count == 1, f"FEHLER: {bl}/{year} hat {count} Einträge"
    
    # Erfolg
    print("Alle 80 Bundesland/Jahr-Kombinationen vorhanden")
    print("TEST 1: PASSED")
    
    # Gib Daten zurück für weitere Tests
    return inkar_long


def test_simulate_soep_data():
    """
    Test 2: Simuliere SOEP-Daten mit syear
    
    Erstellt synthetische SOEP-Daten für alle Jahre.
    """
    # Header
    print("\n--- Test 2: Simuliere SOEP mit syear ---")
    
    # Setze Seed für Reproduzierbarkeit
    np.random.seed(42)
    
    # Anzahl pro Jahr
    n_per_year = 100
    
    # Leere Liste für Daten
    soep_data = []
    
    # Iteriere durch Jahre
    for year in [2015, 2016, 2017, 2018, 2019]:
        # Iteriere durch Personen
        for i in range(n_per_year):
            # Erstelle Eintrag
            soep_data.append({
                'id': len(soep_data) + 1,
                'syear': year,
                'einkommenj1': np.random.uniform(20000, 60000),
                'bildung': np.random.uniform(9, 16)
            })
    
    # Erstelle DataFrame
    df_soep = pd.DataFrame(soep_data)
    
    # Ausgabe
    print(f"Simulierte SOEP-Daten: {len(df_soep)} Zeilen")
    print(f"Jahre: {df_soep['syear'].value_counts().sort_index().to_dict()}")
    
    # Prüfe
    assert len(df_soep) == 500, f"FEHLER: {len(df_soep)} != 500"
    
    # Erfolg
    print("TEST 2: PASSED")
    
    # Gib Daten zurück
    return df_soep


def test_clustering(df_soep):
    """
    Test 3: Clustering
    
    Erstellt 16 Cluster aus den simulierten SOEP-Daten.
    """
    # Header
    print("\n--- Test 3: Clustering ---")
    
    # Erstelle Mapper
    mapper = RegionalClusterMapper(n_clusters=16, random_state=42)
    
    # Führe Clustering durch
    df_soep = mapper.create_clusters(df_soep, ['einkommenj1', 'bildung'])
    
    # Prüfe ob region_cluster Spalte existiert
    assert 'region_cluster' in df_soep.columns, "FEHLER: region_cluster fehlt"
    
    # Prüfe Anzahl Cluster
    n_clusters = df_soep['region_cluster'].nunique()
    assert n_clusters == 16, f"FEHLER: {n_clusters} != 16 Cluster"
    
    # Erstelle Profile
    cluster_profiles = mapper.profile_clusters(df_soep, ['einkommenj1', 'bildung'])
    
    # Prüfe Profile
    assert len(cluster_profiles) == 16, "FEHLER: Nicht 16 Profile"
    
    # Erfolg
    print("TEST 3: PASSED")
    
    # Gib zurück
    return df_soep, cluster_profiles


def test_matching(cluster_profiles):
    """
    Test 4: Matching
    
    Prüft ob alle 16 Bundesländer 1:1 zugeordnet werden.
    """
    # Header
    print("\n--- Test 4: Matching ---")
    
    # Führe Matching durch
    matching = match_clusters_to_bundeslaender(
        cluster_profiles,
        cluster_features=['einkommenj1', 'bildung'],
        inkar_features=['medianeinkommen', 'abiturquote']
    )
    
    # Drucke Report
    print_matching_report(matching)
    
    # Prüfe 1:1 Zuordnung
    assigned = list(matching['mapping'].values())
    n_assigned = len(assigned)
    n_unique = len(set(assigned))
    
    # Ausgabe
    print(f"\nZugeordnete Bundesländer: {n_assigned}")
    print(f"Unique Bundesländer: {n_unique}")
    
    # Prüfe
    assert n_unique == 16, f"FEHLER: Nur {n_unique} unique Bundesländer!"
    
    # Erfolg
    print("TEST 4: PASSED")
    
    # Gib zurück
    return matching


def test_merge_by_year(df_soep, matching, inkar_long):
    """
    Test 5: Jahr-spezifischer Merge
    
    Prüft ob der Merge korrekt funktioniert.
    """
    # Header
    print("\n--- Test 5: Jahr-spezifischer Merge ---")
    
    # Führe Merge durch
    df_final = merge_inkar_by_year(df_soep, matching['mapping'], 'syear')
    
    # Ausgabe
    print(f"Finale Daten: {df_final.shape}")
    print(f"Spalten: {df_final.columns.tolist()}")
    
    # Prüfe Zeilenanzahl
    assert len(df_final) == len(df_soep), "FEHLER: Zeilen verloren!"
    
    # Prüfe INKAR-Spalten
    inkar_cols = ['arbeitslosenquote', 'schulabg_ohne_abschluss', 'abiturquote',
                  'kinderarmut', 'betreuungsquote', 'medianeinkommen']
    for col in inkar_cols:
        assert col in df_final.columns, f"FEHLER: {col} fehlt!"
    
    # Prüfe Kinderarmut 2015 = NaN
    nan_2015 = df_final[df_final['syear'] == 2015]['kinderarmut'].isna().sum()
    total_2015 = len(df_final[df_final['syear'] == 2015])
    print(f"\nKinderarmut 2015: {nan_2015}/{total_2015} NaN (erwartet: alle)")
    assert nan_2015 == total_2015, "FEHLER: Kinderarmut 2015 sollte NaN sein!"
    
    # Prüfe Kinderarmut 2016-2019 nicht NaN
    nan_other = df_final[df_final['syear'] > 2015]['kinderarmut'].isna().sum()
    print(f"Kinderarmut 2016-2019: {nan_other} NaN (erwartet: 0)")
    assert nan_other == 0, "FEHLER: Kinderarmut 2016-2019 sollte nicht NaN sein!"
    
    # Erfolg
    print("TEST 5: PASSED")
    
    # Gib zurück
    return df_final


def test_year_specific_values(df_final, matching, inkar_long):
    """
    Test 6: Jahr-spezifische Werte korrekt
    
    Prüft ob die Jahr-spezifischen INKAR-Werte korrekt zugeordnet sind.
    """
    # Header
    print("\n--- Test 6: Jahr-spezifische Werte ---")
    
    # Wähle ein Bundesland zum Testen
    test_cluster = list(matching['mapping'].keys())[0]
    test_bl = matching['mapping'][test_cluster]
    
    # Info
    print(f"Teste: Cluster {test_cluster} -> {test_bl}\n")
    
    # Vergleiche für jedes Jahr
    for year in [2015, 2016, 2017, 2018, 2019]:
        # SOEP-Wert
        soep = df_final[(df_final['region_cluster'] == test_cluster) & (df_final['syear'] == year)]
        
        # Prüfe ob Daten vorhanden
        if len(soep) > 0:
            # Hole Arbeitslosenquote aus SOEP-Merge
            alq_soep = soep['arbeitslosenquote'].iloc[0]
            
            # Hole Original-INKAR-Wert
            inkar = inkar_long[(inkar_long['bundesland'] == test_bl) & (inkar_long['year'] == year)]
            alq_inkar = inkar['arbeitslosenquote'].iloc[0]
            
            # Vergleiche
            diff = abs(alq_soep - alq_inkar)
            match = "OK" if diff < 0.001 else "FEHLER!"
            
            # Ausgabe
            print(f"  {test_bl} {year}: SOEP={alq_soep:.2f}, INKAR={alq_inkar:.2f} -> {match}")
            
            # Prüfe
            assert diff < 0.001, f"FEHLER: Werte stimmen nicht für {test_bl} {year}!"
    
    # Erfolg
    print("\nTEST 6: PASSED")


def run_integration_tests():
    """
    Führe alle Integrationstests aus
    """
    # Header
    print("=" * 70)
    print("INTEGRATIONSTEST: Vollständiger Workflow")
    print("=" * 70)
    
    # Test 1: INKAR Long-Format
    inkar_long = test_inkar_long_format_completeness()
    
    # Test 2: Simuliere SOEP
    df_soep = test_simulate_soep_data()
    
    # Test 3: Clustering
    df_soep, cluster_profiles = test_clustering(df_soep)
    
    # Test 4: Matching
    matching = test_matching(cluster_profiles)
    
    # Test 5: Jahr-spezifischer Merge
    df_final = test_merge_by_year(df_soep, matching, inkar_long)
    
    # Test 6: Jahr-spezifische Werte
    test_year_specific_values(df_final, matching, inkar_long)
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("INTEGRATIONSTEST ERFOLGREICH")
    print("=" * 70)
    print("- INKAR Long-Format: 80 Zeilen (16 BL × 5 Jahre)")
    print("- Alle 16 Bundesländer 1:1 zugeordnet")
    print("- Jahr-spezifischer Merge funktioniert")
    print("- Kinderarmut 2015 = NaN (korrekt)")
    print("- Jahr-spezifische Werte stimmen überein")
    print("=" * 70)
    
    # Gib True zurück bei Erfolg
    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Führe Tests aus
    success = run_integration_tests()
    
    # Exit Code
    if success:
        print("\nAlle Tests bestanden!")
        exit(0)
    else:
        print("\nTests fehlgeschlagen!")
        exit(1)
