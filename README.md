@"
# AI Assignment 1 - AI_A1_G06

## Project Overview

This project implements an Artificial Intelligence pipeline for the AI_A1_G06 dataset.

The project was tested using:

Operating System: Windows
Python: 3.13.7
pip: 25.2

The project dependencies are specified in requirements.txt.

Main packages
NumPy 2.5.3
pandas 3.0.6
scikit-learn 1.9.1
matplotlib 3.11.2
seaborn 0.13.2
joblib 1.6.0


Environment Setup

Clone the repository:

git clone https://github.com/maramyaser/AI_A1_G06.git
cd AI_A1_G06

python -m venv .env
.\.env\Scripts\Activate


python --version
Python 3.13.7
python -m pip install --upgrade pip
pip install -r requirements.txt

The project dataset is located at:

data/AI_A1_G06.csv

The dataset contains 300 records and 9 columns.

The pipeline verifies:

Number of rows and columns
Missing values
Duplicate record IDs
Feature matrix construction
Dataset consistency



Known Limitations
The models are trained on the provided AI_A1_G06 dataset, so their performance depends on the quality and distribution of this dataset.
Predictions should not be assumed to generalize to populations or datasets with substantially different distributions.
The clustering labels represent mathematically generated groups and do not necessarily correspond to real-world categories.
Model predictions are dependent on the input feature values and the preprocessing used during training.
The classification and regression metrics are based on the available dataset and test split and should not be interpreted as universal real-world performance measures.
Reproducibility requires using the same dataset, Python environment, dependency versions, and project code.
The generated __pycache__ directories and .pyc files are Python runtime files and are not required for reproducing the project.


Expexted outputs:
After successfully running the complete pipeline, the project should contain generated artifacts including:

artifacts/
├── data_report.json
├── classification_metrics.json
├── regression_metrics.json
├── cluster_plot.png
└── ...

Trained models are stored in:
models/

python .\run_all.py --data .\data\AI_A1_G06.csv --output .\artifacts --group AI_A1_G06                                           
Loaded dataset: data\AI_A1_G06.csv
Rows: 300
Columns: 9
Missing values: 0
Duplicate record IDs: 0
Feature matrix shape: (300, 6)
Data report saved to: artifacts\data_report.json
Group code: AI_A1_G06
SHA-256: e011d244db231d7b1f162e7aadcf66f5b72adf70484129e75bbd17854587b099

Regression completed.
Training rows: 240
Testing rows: 60
Test MAE: 41.4432
Test RMSE: 51.6010
Test R²: 0.9806
Metrics saved to: artifacts\regression_metrics.json
Loss plot saved to: artifacts\regression_loss.png
Regression model saved to: models\regression_model.json

Classification completed.
Training rows: 240
Testing rows: 60
Accuracy: 0.6167
Precision: 0.2500
Recall: 0.3846
F1: 0.3030

Confusion Matrix:
[[32 15]
 [ 8  5]]

Metrics saved to: artifacts\classification_metrics.json
Confusion matrix saved to: artifacts\confusion_matrix.png
Classification model saved to: models\classification_model.joblib
k=2 | Silhouette Score: 0.2245
k=3 | Silhouette Score: 0.1732
k=4 | Silhouette Score: 0.1625
k=5 | Silhouette Score: 0.1733

Clustering completed.
Selected k: 2
Selected silhouette score: 0.2245

Cluster counts:
0    149
1    151
Name: count, dtype: int64

Metrics saved to: artifacts\clustering_metrics.json
Clusters saved to: artifacts\clusters.csv
Cluster plot saved to: artifacts\cluster_plot.png
Clustering model saved to: models\clustering_model.joblib

Stage 1, Stage 2, Stage 3, and Stage 4 completed successfully

python .\predict.py --record '{\"plot_area_ha\":1.2,\"rainfall_mm\":81,\"soil_ph\":5.7,\"seed_kg\":210,\"distance_km\":14,\"arrival_hour\":9}'
{"regression_prediction": 567.5203, "classification_prediction": 0, "classification_probability": 0.3816, "cluster_label": 0, "group_code": "AI_A1_G06", "model_version": "1.0"}




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
Umutoniwase Alice    - Data and UX lead
Ishimwe Alphaxad     - Classification enginnering
Muram Yaser          - Reproducibility and release lead
Nsabimana Irene      - regression enginnering
Uwimbabazi Aimerance - clustration enginnering


github repo: https://github.com/maramyaser/AI_A1_G06.git
Final Commit Hash

The final submission commit hash is recorded after the README is finalized.

FINAL_COMMIT_HASH
Final Commit Hash:30b95ae2192a9021343ab59e6dfe63a02962e151
Dataset SHA-256: e011d244db231d7b1f162e7aadcf66f5b72adf70484129e75bbd17854587b099
Python Version Tested: Python 3.13.7