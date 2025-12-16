# PathPredict: Educational Success Forecasting

**Capstone Project - Data Science & AI Bootcamp**

## 🎯 Project Overview

Predicting educational success in Germany by analyzing the impact of household and regional socioeconomic factors using SOEP (German Socio-Economic Panel) data combined with external regional indicators.

## 📊 Data Sources

1. **SOEP Practice Dataset** (2015-2019)
   - ~6,000 individuals
   - Variables: education years, income, household composition, health, life satisfaction
   
2. **External Regional Data** (freely available)
   - INKAR: regional socioeconomic indicators

3. **Synthetic Regional Clustering**
   - K-Means clustering to create 16 regional proxies
   - Based on socioeconomic characteristics

## 🏗️ Project Structure

```
PathPredict_Educational_Success/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Original SOEP files
│   ├── external/               # Regional data from Destatis, etc.
│   └── processed/              # Cleaned, merged datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_regional_clustering.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_visualization.ipynb
├── src/
│   ├── data_loader.py          # Load SOEP data
│   ├── regional_mapper.py      # Create synthetic regions
│   ├── features.py             # Feature engineering
│   ├── models.py               # ML models
│   └── visualizations.py       # Plotting functions
└── dashboard/
    └── app.py                  # Streamlit dashboard
```

## 🎯 Key Objectives

1. **Predict educational attainment** (binary: Abitur+ yes/no)
2. **Identify most impactful factors** (household vs. regional)
3. **Create actionable insights** for policymakers
4. **Build interactive dashboard** for scenario planning

## 📈 Methodology

1. Load SOEP Practice Dataset
2. Create synthetic regional clusters (K-Means on socioeconomic features)
3. Map clusters to real regional average data (unemployment, education spending, etc.)
4. Feature engineering (household + regional factors)
5. Train ML models (Random Forest, XGBoost)
6. Analyze feature importance
7. Build what-if scenario simulator
8. Deploy interactive dashboard

## 🛠️ Tech Stack

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** Streamlit
- **Version Control:** Git/GitHub

## 📋 Timeline

- **Week 1:** Data loading, exploration, clustering
- **Week 2:** Feature engineering, external data integration
- **Week 3:** ML modeling, feature importance analysis
- **Week 4:** Dashboard development, documentation
- **Week 5:** Testing, refinement, presentation prep

## 🎓 Learning Goals

- Handle real-world data limitations creatively
- Integrate multiple data sources
- Build end-to-end ML pipeline
- Create production-ready visualization
- Demonstrate social impact focus

## 📝 Notes

This project uses synthetic regional proxies due to data anonymization in SOEP Practice Dataset. The methodology is transparent and scientifically valid, with architecture designed to seamlessly integrate real geographic data if/when available.

