@"
# AI Assignment 1 - AI_A1_G06

## Project Overview

This project implements an Artificial Intelligence pipeline for the AI_A1_G06 dataset.

The pipeline contains three machine learning tasks:

1. Regression - predict `actual_yield_kg`
2. Classification - predict `dispatch_attention`
3. Clustering - identify groups of records with similar input-feature profiles

The project also provides a prediction script that loads the trained models and produces predictions for a new record.



Dataset:

`data/AI_A1_G06.csv`

Dataset summary:

- Group code: `AI_A1_G06`
- Rows: 300
- Columns: 9
- Input features: 6
- Missing values: 0
- Duplicate record IDs: 0

 Input Features

- `plot_area_ha`
- `rainfall_mm`
- `soil_ph`
- `seed_kg`
- `distance_km`
- `arrival_hour`

Targets

Regression target:

`actual_yield_kg`

Classification target:

`dispatch_attention`

The classification target contains:

- 0: 233 records
- 1: 67 records

This corresponds to approximately:

- 77.67% class 0
- 22.33% class 1


Project Structure

AI_A1_G06/
│
├── data/
│   └── AI_A1_G06.csv
│
├── src/
│   ├── data_pipeline.py
│   ├── regression.py
│   ├── classification.py
│   └── clustering.py
│
├── models/
│   ├── regression_model.json
│   ├── classification_model.joblib
│   └── clustering_model.joblib
│
├── artifacts/
│   ├── data_report.json
│   ├── regression_metrics.json
│   ├── regression_loss.png
│   ├── classification_metrics.json
│   ├── confusion_matrix.png
│   ├── clustering_metrics.json
│   ├── clusters.csv
│   └── cluster_plot.png
│
├── generate_dataset.py
├── run_all.py
├── predict.py
├── requirements.txt
└── README.md


members:
                     - Data and UX lead
                     - Regression engineer
Muram Yaser 25/28164 - 3&4 classification enginnering, Clustering and QAengineer
                     -  5 Reproducibility and release lead


github repo: https://github.com/maramyaser/AI_A1_G06.git
