# Notebooks

This directory contains Jupyter notebooks for exploring and analyzing the Fairness-Aware Music Recommender system.

## Available Notebooks

### 1. `quick_demo.ipynb`
A quick demonstration of the system that shows:
- Data loading and processing
- Model training
- Bias detection
- Fairness-aware re-ranking
- Comparison of baseline vs fair recommendations

### 2. `data_exploration.ipynb`
Comprehensive data exploration and analysis including:
- Dataset statistics and distributions
- Popularity bias analysis
- Genre diversity analysis
- User behavior patterns
- Bias detection in recommendations
- Fairness improvement demonstrations

## How to Run

1. Install Jupyter notebook:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open the desired notebook and run all cells

## Requirements

Make sure you have processed the data first by running:
```bash
python ../src/data_processing.py
```

Or train the models:
```bash
python ../src/train_models.py
```






