"""
Data Loader Module
==================
Handles loading and initial processing of SOEP Practice Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


class SOEPDataLoader:
    """Load and preprocess SOEP Practice Dataset"""
    
    def __init__(self, data_path):
        """
        Initialize loader
        
        Parameters:
        -----------
        data_path : str or Path
            Path to SOEP .dta file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Load SOEP data from STATA file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_stata(self.data_path)
        print(f"✓ Loaded {len(self.df)} observations with {len(self.df.columns)} variables")
        return self.df
    
    def get_basic_info(self):
        """Display basic dataset information"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("SOEP PRACTICE DATASET - BASIC INFORMATION")
        print("="*60)
        
        print(f"\nShape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Time period: {self.df['syear'].min()} - {self.df['syear'].max()}")
        print(f"Unique individuals: {self.df['id'].nunique()}")
        
        print("\n--- Available Variables ---")
        print(self.df.columns.tolist())
        
        print("\n--- Missing Data Summary ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_summary = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        })
        print(missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False))
        
        return self.df.info()
    
    def clean_data(self):
        """Basic data cleaning"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nCleaning data...")
        original_rows = len(self.df)
        
        # Create copy to avoid modifying original
        df_clean = self.df.copy()
        
        # Remove rows with missing education (our target variable)
        df_clean = df_clean.dropna(subset=['bildung'])
        
        # Create age variable if not exists
        if 'alter' not in df_clean.columns and 'gebjahr' in df_clean.columns:
            df_clean['alter'] = df_clean['syear'] - df_clean['gebjahr']
        
        # Filter to adults (18+) for educational attainment analysis
        if 'alter' in df_clean.columns:
            df_clean = df_clean[df_clean['alter'] >= 18]
        
        print(f"✓ Cleaned: {original_rows} → {len(df_clean)} observations")
        print(f"  Removed {original_rows - len(df_clean)} rows")
        
        self.df = df_clean
        return self.df
    
    def create_target_variable(self):
        """
        Create binary target variable for educational attainment
        
        Target: high_education (Abitur+ = 13+ years of education)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df['high_education'] = (self.df['bildung'] >= 13).astype(int)
        
        print("\n--- Target Variable Created ---")
        print(f"high_education (Abitur+):")
        print(self.df['high_education'].value_counts())
        print(f"\nProportion with Abitur+: {self.df['high_education'].mean():.2%}")
        
        return self.df
    
    def get_summary_statistics(self):
        """Generate summary statistics for key variables"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        key_vars = ['bildung', 'alter', 'einkommenj1', 'einkommenm1', 
                    'anz_pers', 'anz_kind', 'gesund_org', 'lebensz_org']
        
        available_vars = [v for v in key_vars if v in self.df.columns]
        
        print("\n--- Summary Statistics ---")
        print(self.df[available_vars].describe())
        
        return self.df[available_vars].describe()
    
    def save_processed_data(self, output_path):
        """Save processed data"""
        if self.df is None:
            raise ValueError("No data to save. Load and process data first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to {output_path}")


def load_soep_quick(file_path):
    """
    Quick load function for SOEP data
    
    Parameters:
    -----------
    file_path : str
        Path to SOEP .dta file
        
    Returns:
    --------
    pd.DataFrame
        Loaded SOEP data
    """
    loader = SOEPDataLoader(file_path)
    df = loader.load_data()
    loader.get_basic_info()
    return df


if __name__ == "__main__":
    # Example usage
    print("SOEP Data Loader Module")
    print("Usage:")
    print("  from src.data_loader import SOEPDataLoader")
    print("  loader = SOEPDataLoader('data/raw/practice_dataset_eng.dta')")
    print("  df = loader.load_data()")
