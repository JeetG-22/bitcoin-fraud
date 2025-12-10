# Machine Learning Project: Bitcoin Fraud Detection

### Create Python Virtual Environment

We use Python virtual enviroments to keep pip packages consistent and localized:

```shell
python3 -m venv venv
# If on Windows CMD, run .\venv\Scripts\activate.bat instead of the following
source venv/bin/activate
pip install -r requirements.txt
```

### Running

Run the following files in the specified order:

1. `elliptic_dataset_download.py`
   - Download dataset from online to local machine.
1. `notebooks/01_eda.ipynb` and `notebooks/01++_eda.ipynb` (Optional)
   - Basic exploratory analysis.
1. `notebooks/02_preprocessing.ipynb`
   - Combines labels and features into joint dataset.
   - Assigns column names to unnamed features.
   - Creates reproducible train-test split.
1. `notebooks/03_visualizations.ipynb`
   - Calculates and visualizes 2D projections using PCA and UMAP.
   - Cumulative explained variance from PCA components.
1. `notebooks/04_supervised_models.ipynb`
   - Trains a Random Forest and Logistic Regression model.
   - Performs hyperparameter tuning with 5-fold cross-validation.
   - Calculates evaluation metrics (f1, ROC-AUC, accuracy, etc).
   - Visualizes incorrect predictions on PCA/UMAP projections.
1. `notebooks/05_evaluation.ipynb` and `notebooks/
   - Unused (evaluation is done within the notebooks themselves)
1. `notebooks/07_isolation_forest.ipynb`
   - Trained and evaluated isolation forest models
1. `notebooks/08_graphCL.ipynb`
   - Trained and evaluated graph contrastive learning models.
