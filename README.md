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
Roadmap

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
- [ ] Model improvements
Author

Atharv Singh
