# Calorie Expenditure Prediction Project

This project predicts calorie expenditure based on various physical and personal metrics using machine learning. It includes data analysis, model training, and a Streamlit web interface for predictions.

## Project Structure

```
Predict-Calaories-Exp/
├── data/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
│   └── sample_submission.csv
├── model/
│   ├── XGBRegressor_best_model.pkl    # Trained XGBoost model
│   └── DecisionTree_best_model.pkl    # Trained Decision Tree model
├── notebooks/
│   └── predict-calorie-expenditure-eda-ml.ipynb  # Data analysis and exploration
├── src/
│   ├── preprocess.py       # Data preprocessing utilities
│   ├── train.py            # Model training script
│   └── streamlit.py        # Web interface
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Features

- Exploratory Data Analysis (EDA) with visualizations
- Multiple machine learning models (XGBoost, Decision Tree)
- Hyperparameter tuning and model evaluation
- Web interface for making predictions
- Data preprocessing pipeline

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Predict-Calaories-Exp
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
```bash
python src/train.py
```

### Running the Web Interface
```bash
streamlit run src/streamlit.py
```

Then open your browser to `http://localhost:8501`

## Model Performance

After training, the following performance metrics were achieved on the validation set:

| Model | MSE (Lower is better) |
|-------|----------------------|
| Decision Tree Regressor | [MSE value will appear after training] |
| XGBoost Regressor | [MSE value will appear after training] |

To see the exact MSE values, run the training script:
```bash
python train.py
```

The models are automatically saved in the `model/` directory after training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
