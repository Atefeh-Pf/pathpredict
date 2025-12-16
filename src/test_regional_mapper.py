"""
Unit Tests für regional_mapper.py
==================================
Führe aus mit: python test_regional_mapper.py
"""

# Importiere Test-Framework
import unittest

# Importiere Pandas für DataFrames
import pandas as pd

# Importiere NumPy für numerische Operationen
import numpy as np

# Importiere zu testende Funktionen
from regional_mapper import (
    create_inkar_long_format,
    create_inkar_averaged,
    calculate_match_quality,
    calculate_quality_statistics,
    match_clusters_to_bundeslaender,
    merge_inkar_by_year,
    RegionalClusterMapper
)


class TestInkarLongFormat(unittest.TestCase):
    """Tests für create_inkar_long_format()"""
    
    def setUp(self):
        """Erstelle Long-Format vor jedem Test"""
        # Hole Long-Format Daten
        self.df = create_inkar_long_format()
    
    def test_shape(self):
        """Test: 80 Zeilen (16 Bundesländer × 5 Jahre)"""
        # Prüfe Anzahl Zeilen
        self.assertEqual(self.df.shape[0], 80)
    
    def test_columns(self):
        """Test: Alle erwarteten Spalten vorhanden"""
        # Erwartete Spalten
        expected = [
            'bundesland', 'year', 'arbeitslosenquote', 
            'schulabg_ohne_abschluss', 'abiturquote', 
            'kinderarmut', 'betreuungsquote', 'medianeinkommen'
        ]
        # Prüfe jede Spalte
        for col in expected:
            self.assertIn(col, self.df.columns)
    
    def test_years(self):
        """Test: Jahre 2015-2019 vorhanden"""
        # Hole unique Jahre
        years = set(self.df['year'].unique())
        # Erwartete Jahre
        expected = {2015, 2016, 2017, 2018, 2019}
        # Prüfe
        self.assertEqual(years, expected)
    
    def test_bundeslaender_count(self):
        """Test: 16 Bundesländer vorhanden"""
        # Zähle unique Bundesländer
        count = self.df['bundesland'].nunique()
        # Prüfe
        self.assertEqual(count, 16)
    
    def test_kinderarmut_2015_nan(self):
        """Test: Kinderarmut 2015 ist NaN"""
        # Filtere Jahr 2015
        df_2015 = self.df[self.df['year'] == 2015]
        # Zähle NaN
        nan_count = df_2015['kinderarmut'].isna().sum()
        # Alle 16 sollten NaN sein
        self.assertEqual(nan_count, 16)
    
    def test_kinderarmut_2016_2019_not_nan(self):
        """Test: Kinderarmut 2016-2019 ist nicht NaN"""
        # Filtere Jahre > 2015
        df_other = self.df[self.df['year'] > 2015]
        # Zähle NaN
        nan_count = df_other['kinderarmut'].isna().sum()
        # Keine sollten NaN sein
        self.assertEqual(nan_count, 0)
    
    def test_each_combination_exists(self):
        """Test: Jede Bundesland/Jahr-Kombination existiert genau einmal"""
        # Liste der Bundesländer
        bundeslaender = self.df['bundesland'].unique()
        # Liste der Jahre
        years = [2015, 2016, 2017, 2018, 2019]
        # Prüfe jede Kombination
        for bl in bundeslaender:
            for year in years:
                # Zähle Einträge
                count = len(self.df[(self.df['bundesland'] == bl) & (self.df['year'] == year)])
                # Genau einer
                self.assertEqual(count, 1, f"{bl}/{year} hat {count} Einträge")


class TestBundeslaender(unittest.TestCase):
    """Tests für Bundesländer-Vollständigkeit"""
    
    def test_all_bundeslaender_present(self):
        """Test: Alle 16 Bundesländer in INKAR-Daten"""
        # Erwartete Bundesländer
        expected = [
            'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
            'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg',
            'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
            'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen'
        ]
        # Hole Long-Format
        df = create_inkar_long_format()
        # Hole actual Bundesländer
        actual = df['bundesland'].unique().tolist()
        # Prüfe jeden
        for bl in expected:
            self.assertIn(bl, actual, f"Bundesland '{bl}' fehlt")
        # Prüfe Anzahl
        self.assertEqual(len(actual), 16)


class TestMatchQuality(unittest.TestCase):
    """Tests für calculate_match_quality()"""
    
    def test_sehr_gut_lower_bound(self):
        """Test: 0.0 ist SEHR_GUT"""
        self.assertEqual(calculate_match_quality(0.0), 'SEHR_GUT')
    
    def test_sehr_gut_upper_bound(self):
        """Test: 0.49 ist SEHR_GUT"""
        self.assertEqual(calculate_match_quality(0.49), 'SEHR_GUT')
    
    def test_gut_lower_bound(self):
        """Test: 0.5 ist GUT"""
        self.assertEqual(calculate_match_quality(0.5), 'GUT')
    
    def test_gut_upper_bound(self):
        """Test: 0.99 ist GUT"""
        self.assertEqual(calculate_match_quality(0.99), 'GUT')
    
    def test_mittel_lower_bound(self):
        """Test: 1.0 ist MITTEL"""
        self.assertEqual(calculate_match_quality(1.0), 'MITTEL')
    
    def test_mittel_upper_bound(self):
        """Test: 1.49 ist MITTEL"""
        self.assertEqual(calculate_match_quality(1.49), 'MITTEL')
    
    def test_schwach_lower_bound(self):
        """Test: 1.5 ist SCHWACH"""
        self.assertEqual(calculate_match_quality(1.5), 'SCHWACH')
    
    def test_schwach_high_value(self):
        """Test: 5.0 ist SCHWACH"""
        self.assertEqual(calculate_match_quality(5.0), 'SCHWACH')


class TestQualityStatistics(unittest.TestCase):
    """Tests für calculate_quality_statistics()"""
    
    def setUp(self):
        """Erstelle Test-DataFrame"""
        # Test-Daten: 2 von jeder Qualitätsstufe
        self.test_df = pd.DataFrame({
            'cluster_id': range(8),
            'bundesland': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'distanz': [0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6, 2.0],
            'quality': ['SEHR_GUT', 'SEHR_GUT', 'GUT', 'GUT', 
                       'MITTEL', 'MITTEL', 'SCHWACH', 'SCHWACH']
        })
        # Berechne Statistiken
        self.stats = calculate_quality_statistics(self.test_df)
    
    def test_counts(self):
        """Test: Counts pro Qualitätsstufe"""
        self.assertEqual(self.stats['counts']['SEHR_GUT'], 2)
        self.assertEqual(self.stats['counts']['GUT'], 2)
        self.assertEqual(self.stats['counts']['MITTEL'], 2)
        self.assertEqual(self.stats['counts']['SCHWACH'], 2)
    
    def test_percentages(self):
        """Test: Prozentsätze korrekt"""
        # Jede Kategorie hat 2/8 = 25%
        self.assertEqual(self.stats['percentages']['SEHR_GUT'], 25.0)
        self.assertEqual(self.stats['percentages']['GUT'], 25.0)
    
    def test_passed(self):
        """Test: Bestanden wenn >= 50% GUT oder besser"""
        # 4/8 = 50% -> sollte bestanden sein
        self.assertTrue(self.stats['passed'])
    
    def test_good_matches(self):
        """Test: Anzahl guter Matches"""
        # 2 SEHR_GUT + 2 GUT = 4
        self.assertEqual(self.stats['good_matches'], 4)


class TestMergeByYear(unittest.TestCase):
    """Tests für merge_inkar_by_year()"""
    
    def setUp(self):
        """Erstelle Test-Daten"""
        # Test SOEP-DataFrame
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'region_cluster': [0, 0, 1, 1, 2],
            'syear': [2015, 2019, 2016, 2018, 2017]
        })
        # Test Mapping
        self.test_mapping = {
            0: 'Bayern',
            1: 'Berlin',
            2: 'Hamburg'
        }
    
    def test_row_count_preserved(self):
        """Test: Alle Zeilen erhalten"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Prüfe Zeilenanzahl
        self.assertEqual(len(result), 5)
    
    def test_bundesland_correct(self):
        """Test: Bundesland korrekt zugeordnet"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Prüfe erste Zeile (Cluster 0 -> Bayern)
        self.assertEqual(result.iloc[0]['matched_bundesland'], 'Bayern')
        # Prüfe dritte Zeile (Cluster 1 -> Berlin)
        self.assertEqual(result.iloc[2]['matched_bundesland'], 'Berlin')
    
    def test_inkar_columns_added(self):
        """Test: INKAR-Spalten hinzugefügt"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Prüfe Spalten
        self.assertIn('arbeitslosenquote', result.columns)
        self.assertIn('abiturquote', result.columns)
        self.assertIn('kinderarmut', result.columns)
    
    def test_kinderarmut_2015_nan(self):
        """Test: Kinderarmut 2015 ist NaN"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Filtere 2015
        row_2015 = result[result['syear'] == 2015]
        # Prüfe NaN
        self.assertTrue(row_2015['kinderarmut'].isna().all())
    
    def test_kinderarmut_2019_not_nan(self):
        """Test: Kinderarmut 2019 ist nicht NaN"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Filtere 2019
        row_2019 = result[result['syear'] == 2019]
        # Prüfe nicht NaN
        self.assertFalse(row_2019['kinderarmut'].isna().any())
    
    def test_year_specific_values_differ(self):
        """Test: Jahr-spezifische Werte unterscheiden sich"""
        # Führe Merge durch
        result = merge_inkar_by_year(self.test_df, self.test_mapping, 'syear')
        # Hole Bayern 2015 und 2019
        bayern_2015 = result[(result['syear'] == 2015) & (result['matched_bundesland'] == 'Bayern')]
        bayern_2019 = result[(result['syear'] == 2019) & (result['matched_bundesland'] == 'Bayern')]
        # Arbeitslosenquote sollte unterschiedlich sein
        alq_2015 = bayern_2015['arbeitslosenquote'].values[0]
        alq_2019 = bayern_2019['arbeitslosenquote'].values[0]
        self.assertNotEqual(alq_2015, alq_2019)


class TestMatchingOutput(unittest.TestCase):
    """Tests für match_clusters_to_bundeslaender() Output-Struktur"""
    
    def setUp(self):
        """Erstelle minimale Test-Daten"""
        # Cluster-Profile
        self.cluster_profiles = pd.DataFrame({
            'einkommenj1': [30000, 40000, 50000],
            'bildung': [11, 13, 15]
        }, index=[0, 1, 2])
    
    def test_output_has_mapping(self):
        """Test: Output enthält 'mapping'"""
        # Führe Matching durch (mit echten INKAR-Daten)
        # Wir mocken hier nicht, sondern nutzen subset
        results = match_clusters_to_bundeslaender(self.cluster_profiles)
        self.assertIn('mapping', results)
    
    def test_output_has_results(self):
        """Test: Output enthält 'results'"""
        results = match_clusters_to_bundeslaender(self.cluster_profiles)
        self.assertIn('results', results)
    
    def test_output_has_quality(self):
        """Test: Output enthält 'quality'"""
        results = match_clusters_to_bundeslaender(self.cluster_profiles)
        self.assertIn('quality', results)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Führe alle Tests aus
    print("=" * 60)
    print("UNIT TESTS FÜR REGIONAL_MAPPER.PY")
    print("=" * 60)
    
    # Starte Tests
    unittest.main(verbosity=2)
