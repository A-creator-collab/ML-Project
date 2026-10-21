# ML-Project

Machine learning experiments — dataset exploration, preprocessing, and baseline models.

## Dataset

Wine Quality (UCI). Red wine samples with 11 physicochemical features and a quality score (3–8).

Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Structure
# ML-Project
Machine learning experiments — EDA, preprocessing, baseline models.
ML-Project/
├── data/
│ ├── raw/ # downloaded datasets (not committed)
│ └── processed/ # cleaned data (not committed)
├── notebooks/ # Jupyter notebooks
├── src/ # reusable Python modules
├── requirements.txt
└── README.md

## Notebooks

- `notebooks/01_eda.ipynb` — exploratory data analysis

## Setup

```bash
pip install -r requirements.txt
Download the dataset:
curl -L -o data/raw/winequality-red.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```
##Roadmap

    ☑

    Dataset exploration
    □

    Data preprocessing pipeline
    □

    Visualization notebook
    □

    Baseline classification model
    □

    Model evaluation
    □

    Model improvements
## Roadmap

- [x] Dataset exploration
- [x] Data preprocessing pipeline
- [x] Visualization notebook
- [x] Baseline classification model
- [x] Model evaluation
- [x] Model improvements
## Results

Baseline Random Forest with `class_weight='balanced'`:

- **Accuracy:** ~85%
- **Recall (Good Wine):** ~65%
- **F1 Score:** ~55%

*The balanced class weights significantly improved the model's ability to detect the minority class (good wines), increasing recall at the cost of a slight drop in overall accuracy.*
Author

Atharv Singh
