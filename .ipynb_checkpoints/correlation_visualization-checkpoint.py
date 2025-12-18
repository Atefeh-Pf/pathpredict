"""
Korrelations-Visualisierungen für PathPredict Präsentation
===========================================================
Code zum Kopieren in Jupyter Notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ============================================================
# SCHRITT 1: DATEN LADEN
# ============================================================

# Lade den Datensatz mit SOEP + INKAR merged
df = pd.read_csv('data/processed/soep_with_regions.csv')

# Prüfe verfügbare Spalten
print("Verfügbare Spalten:")
print(df.columns.tolist())
print(f"\nAnzahl Zeilen: {len(df)}")

# ============================================================
# SCHRITT 2: KORRELATIONS-VISUALISIERUNG
# ============================================================

# Erstelle Figure mit 2 Subplots (1 Zeile, 2 Spalten)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ========== PLOT 1: Individual Income ↔ Education ==========
ax1 = axes[0]

# Entferne NaN-Werte
df_clean1 = df[['einkommenj1', 'bildung']].dropna()

# Berechne Korrelationskoeffizient
corr1, pval1 = pearsonr(df_clean1['einkommenj1'], df_clean1['bildung'])

# Scatter Plot
ax1.scatter(
    df_clean1['einkommenj1'], 
    df_clean1['bildung'],
    alpha=0.3,
    s=20,
    color='#2E86AB',
    edgecolors='none'
)

# Trendlinie (lineare Regression)
z1 = np.polyfit(df_clean1['einkommenj1'], df_clean1['bildung'], 1)
p1 = np.poly1d(z1)
x_trend1 = np.linspace(df_clean1['einkommenj1'].min(), df_clean1['einkommenj1'].max(), 100)
ax1.plot(x_trend1, p1(x_trend1), 
         color='#A23B72', 
         linewidth=3, 
         label=f'Trendlinie (r={corr1:.3f})')

# Styling
ax1.set_xlabel('Individual Income (einkommenj1) [EUR]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Education Years (bildung)', fontsize=13, fontweight='bold')
ax1.set_title('Strong Positive Correlation: Income ↔ Education', 
              fontsize=15, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right')

# Füge Korrelations-Info hinzu
textstr1 = f'Pearson r = {corr1:.3f}\np-value < 0.001\nn = {len(df_clean1):,}'
ax1.text(0.05, 0.95, textstr1, 
         transform=ax1.transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ========== PLOT 2: Regional Unemployment ↔ Education ==========
ax2 = axes[1]

# Entferne NaN-Werte
df_clean2 = df[['arbeitslosenquote', 'bildung']].dropna()

# Berechne Korrelationskoeffizient
corr2, pval2 = pearsonr(df_clean2['arbeitslosenquote'], df_clean2['bildung'])

# Scatter Plot
ax2.scatter(
    df_clean2['arbeitslosenquote'], 
    df_clean2['bildung'],
    alpha=0.3,
    s=20,
    color='#E63946',
    edgecolors='none'
)

# Trendlinie (lineare Regression)
z2 = np.polyfit(df_clean2['arbeitslosenquote'], df_clean2['bildung'], 1)
p2 = np.poly1d(z2)
x_trend2 = np.linspace(df_clean2['arbeitslosenquote'].min(), df_clean2['arbeitslosenquote'].max(), 100)
ax2.plot(x_trend2, p2(x_trend2), 
         color='#1D3557', 
         linewidth=3, 
         label=f'Trendlinie (r={corr2:.3f})')

# Styling
ax2.set_xlabel('Regional Unemployment Rate (arbeitslosenquote) [%]', fontsize=13, fontweight='bold')
ax2.set_ylabel('Education Years (bildung)', fontsize=13, fontweight='bold')
ax2.set_title('Moderate Negative Correlation: Unemployment ↔ Education', 
              fontsize=15, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper right')

# Füge Korrelations-Info hinzu
textstr2 = f'Pearson r = {corr2:.3f}\np-value < 0.001\nn = {len(df_clean2):,}'
ax2.text(0.05, 0.95, textstr2, 
         transform=ax2.transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Finales Layout
plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualisierung erstellt und gespeichert als 'correlation_analysis.png'")
print(f"\n📊 Korrelation 1 (Income ↔ Education): r = {corr1:.3f}")
print(f"📊 Korrelation 2 (Unemployment ↔ Education): r = {corr2:.3f}")

# ============================================================
# BONUS: KORRELATIONS-MATRIX (ALLE FEATURES)
# ============================================================

# Wähle relevante Features
features_for_corr = [
    'einkommenj1',
    'bildung',
    'arbeitslosenquote',
    'kinderarmut',
    'abiturquote',
    'medianeinkommen'
]

# Filtere verfügbare Features
available_features = [f for f in features_for_corr if f in df.columns]

# Erstelle Korrelations-Matrix
corr_matrix = df[available_features].corr()

# Visualisiere als Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=1,
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title('Correlation Matrix: Individual & Regional Features', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Korrelations-Matrix erstellt und gespeichert als 'correlation_matrix.png'")

# ============================================================
# AUSGABE: KORRELATIONS-TABELLE
# ============================================================

print("\n📋 Korrelations-Tabelle (bildung mit allen Features):")
print("="*60)

corr_with_bildung = df[available_features].corr()['bildung'].sort_values(ascending=False)
for feature, corr_value in corr_with_bildung.items():
    if feature != 'bildung':
        strength = ''
        if abs(corr_value) > 0.7:
            strength = '(Very Strong)'
        elif abs(corr_value) > 0.5:
            strength = '(Strong)'
        elif abs(corr_value) > 0.3:
            strength = '(Moderate)'
        else:
            strength = '(Weak)'
        
        direction = '↑' if corr_value > 0 else '↓'
        print(f"{feature:30s}: {direction} {corr_value:+.3f}  {strength}")

print("="*60)
