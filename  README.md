# Computational Intelligence for Movie Recommendation Systems

A comprehensive implementation of multiple Computational Intelligence paradigms applied to collaborative filtering for the MovieLens 1M dataset.

## Project Overview

This project explores and compares four different CI approaches for movie recommendations:
- **Neural Collaborative Filtering (NCF)** - Artificial Neural Network
- **Hybrid NCF** - Deep Neural Network with content features
- **PSO-Optimized NCF** - Evolutionary Algorithm optimization
- **ANFIS** - Adaptive Neuro-Fuzzy Inference System

## Results

| Model | Test RMSE | Improvement | Paradigm |
|-------|-----------|-------------|----------|
| **ANFIS** | **0.8681** | **+3.72%** | Fuzzy Systems + ANN |
| PSO-NCF | 0.8721 | +3.27% | Evolutionary Algorithm |
| Hybrid NCF | 0.8920 | +1.07% | Deep Neural Network |
| NCF Baseline | 0.9016 | baseline | Neural Network |

### Key Findings
-  **ANFIS achieved best performance** with 3.72% improvement over baseline
- **PSO optimization** effectively tuned hyperparameters (+3.27%)
- **Content features** provided modest improvement (+1.07%)
- **All models** demonstrated good generalization (minimal overfitting)

## Dataset

**MovieLens 1M**
- 1,000,209 ratings
- 6,040 users
- 3,883 movies
- Rating scale: 1-5 stars
- User demographics included
- link: https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
32GB RAM (recommended)
```

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MovieLens-CI-Recommender.git
cd MovieLens-CI-Recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
1. Download MovieLens 1M dataset from [Kaggle](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset)
2. Extract files to `data/raw/`:
   - `ratings.dat`
   - `movies.dat`
   - `users.dat`

## 🚀 Quick Start

### 1. Preprocess Data
```bash
python main_preprocessing.py
```

### 2. Train Models

#### NCF Baseline
```bash
python training/train_ncf.py
```

#### Hybrid NCF
```bash
python training/train_hybrid_ncf.py
```

#### PSO Optimization
```bash
python training/train_pso.py
```

#### ANFIS
```bash
python training/train_anfis.py
```

### 3. Evaluate and Compare
```bash
python evaluation/compare_models.py
```

## 📐 Model Architectures

### 1. Neural Collaborative Filtering (NCF)
- **Paradigm**: Artificial Neural Network
- **Architecture**: Embedding layers + MLP
- **Parameters**: 1.4M
- **Training**: 13 epochs, 1.9 minutes

### 2. Hybrid NCF
- **Paradigm**: Deep Neural Network
- **Architecture**: GMF + MLP + Content features
- **Parameters**: 1.6M
- **Training**: 15 epochs, 5.7 minutes
- **Features**: 28 user features, 24 movie features

### 3. PSO-Optimized NCF
- **Paradigm**: Evolutionary Algorithm (Swarm Intelligence)
- **Optimization**: Hyperparameter tuning via PSO
- **Parameters**: 1.6M
- **Training**: 125.7 minutes ( Current dev setup:4060RTX + 32GB RAM + Ryzen 7500f)
- **Optimization**: Population=20, Generations=15

### 4. ANFIS
- **Paradigm**: Fuzzy Systems + Neural Learning
- **Architecture**: 5 fuzzy inputs, Takagi-Sugeno inference
- **Parameters**: 576 (highly efficient!)
- **Training**: 50 epochs, 17.1 minutes
- **Unique**: Interpretable fuzzy rules

## Performance Comparison

### RMSE Progression
```
0.9016 (NCF) → 0.8920 (Hybrid) → 0.8721 (PSO) → 0.8681 (ANFIS)
   ↓            ↓                  ↓              ↓
Baseline     +1.07%            +3.27%         +3.72%
```

### Computational Efficiency
- **Most Parameters**: PSO/Hybrid (1.6M)
- **Fewest Parameters**: ANFIS (576) - **2,739x fewer!**
- **Fastest Training**: NCF (1.9 min)
- **Best Accuracy**: ANFIS (0.8681 RMSE)

## 🏗Technical Implementation

### Technologies
- **Framework**: PyTorch 2.1.0
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Optimization**: DEAP (for PSO)
- **Fuzzy Logic**: scikit-fuzzy

### Hardware
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **CPU**: AMD Ryzen 7 7500F
- **RAM**: 32GB
- **Training Time (Total)**: ~6 minutes

## Evaluation Metrics

- **RMSE** (Root Mean Square Error) - Primary metric
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Square Error)
- **Generalization Gap** (Val RMSE - Train RMSE)
- **Training Efficiency** (Time per epoch)

##  Project Structure
```
MovieLens-CI-Recommender/
├── src/              # Core preprocessing and utilities
├── models/           # Model architectures
├── training/         # Training scripts
├── evaluation/       # Evaluation and visualization
├── data/             # Dataset (not included in repo)
└── outputs/          # Model checkpoints and results
```

## Academic Context

This project was developed for **TCI6313 Computational Intelligence** course at MMU Malaysia, demonstrating practical applications of multiple CI paradigms:
1. Artificial Neural Networks
2. Deep Neural Networks
3. Evolutionary Computation
4. Fuzzy Systems



## License

MIT License - see [LICENSE](LICENSE) file



## Acknowledgments

- MovieLens dataset by GroupLens Research
- Course instructors: Prof. Dr. Tan Shing Chiang & Mr. Brendan Hong Jun Zhi
- PyTorch and scikit-learn communities
⭐ **Star this repo if you found it helpful!**