# Iris Classification Project

## Overview
The Iris Classification Project aims to classify iris flowers into three species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*, using four features: sepal length, sepal width, petal length, and petal width. The project utilizes the Iris dataset to train and evaluate multiple machine learning models, optimizing their performance through hyperparameter tuning. Recall is prioritized as the primary metric to ensure accurate species identification, crucial for applications in botany and research. The tuned Random Forest model was selected for its 100% recall and robust performance.

## Key Features
- Data Loading & Preprocessing: Loads and cleans the Iris dataset, splitting it into features and target variables.
- Exploratory Data Analysis (EDA): Visualizes data distributions, feature relationships, and correlations.
- Model Training & Evaluation: Trains and evaluates seven classification models.
- Hyperparameter Tuning: Uses GridSearchCV and RandomizedSearchCV for model optimization.
- Model Prediction: Employs the best-performing model for predicting iris species.

## Key Libraries
The project uses several Python libraries for data processing, visualization, and machine learning. Below are the main libraries:

### Visualization
```python
import matplotlib.pyplot as plt  # Plotting histograms, scatter plots, and other visualizations
import seaborn as sns  # Enhanced visualizations like heatmaps and pair plots

```
### Machine Learning - Preprocessing
```python
from sklearn.preprocessing import LabelEncoder  # Encoding categorical target variables
```
### Machine Learning - Model Selection
```python
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
from sklearn.model_selection import GridSearchCV  # Exhaustive hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV  # Efficient hyperparameter tuning
from sklearn.model_selection import RepeatedStratifiedKFold  # Cross-validation with stratified sampling
```
### Machine Learning - Evaluation Metrics
```python
from sklearn.metrics import confusion_matrix  # Visualizing model errors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Performance metrics
from sklearn.metrics import classification_report  # Detailed performance reports
```
### Machine Learning - Models
```python
from sklearn.linear_model import LogisticRegression  # Logistic regression classification
from sklearn.tree import DecisionTreeClassifier  # Decision tree classification
from sklearn.ensemble import RandomForestClassifier  # Random forest classification
from sklearn.svm import SVC  # Support vector machine classification
from sklearn.neural_network import MLPClassifier  # Neural network classification
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classification
import xgboost as xgb  # Extreme gradient boosting classification
```
###Warnings
```python
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings for cleaner output
```

###Installation
To run the project, ensure Python 3.7+ is installed. Install the required libraries using pip:
```python
To run the project, ensure Python 3.7+ is installed. Install the required libraries using pip:
```

### Usage
Load the Dataset
Download the Iris dataset from the provided source and load it into a pandas DataFrame:
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_task1/main/Iris.csv")
```
Exploratory Data Analysis (EDA)
Visualize feature distributions with histograms:

```python
df.hist(figsize=(10, 8))
plt.show()
```
Explore feature relationships with scatter plots:
```python
sns.pairplot(df, hue='Species')
plt.show()
```
Analyze feature correlations using a heatmap:
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```
### Data Preprocessing
```python
df = df.iloc[:, 1:]
```
Check for duplicates and missing values:
```python
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum()}")
```
Split data into features (X) and target (y):
```python
Split data into features (X) and target (y):
```
###Model Training & Evaluation

Train models using the following example for Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import confusion_matrix, classification_report
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
### Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}
rf_random_search = RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions=param_dist_rf, n_iter=10, cv=5)
rf_random_search.fit(X_train, y_train)
```
Model Prediction
Use the tuned Random Forest model to predict iris species:
```python
Category_RF = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
x_rf = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
x_rf_prediction = rf_random_search.best_estimator_.predict(x_rf)
print(f"Predicted species: {Category_RF[int(x_rf_prediction[0])]}")  # Output: Iris-Setosa
```

# Results

Although all models achieved 100% recall on the test set, the **Random Forest classifier** was selected due to its robustness, ability to handle complex relationships between features, and ease of interpretability. The tuning process further optimized the model, ensuring balanced performance across secondary metrics such as precision, accuracy, and F1-score.

## Model Performance

Seven models were implemented and evaluated, with their test set recall percentages as follows:

- **Logistic Regression**: 100% recall
- **Decision Tree (tuned)**: 100% recall
- **Random Forest (tuned)**: 100% recall
- **Support Vector Machine (SVM)**: 100% recall
- **XGBoost**: 100% recall
- **Naive Bayes (tuned)**: 100% recall
- **Neural Network (tuned)**: 100% recall

Despite all models achieving 100% recall, the **tuned Random Forest model** was selected for its robustness, interpretability, and balanced performance across secondary metrics such as **precision**, **accuracy**, and **F1-score**.

## Hyperparameter Tuning

- **GridSearchCV**: Used for **Naive Bayes** due to its small parameter space (e.g., tuning `var_smoothing`).
- **RandomizedSearchCV**: Applied to other models for efficiency with larger parameter spaces (e.g., `n_estimators`, `max_depth` for Random Forest).

Tuning improved training performance and helped prevent overfitting. However, due to the simplicity of the dataset, test set metrics remained near-perfect.

## Notes

- **Recall Focus**: Recall was prioritized to ensure all true iris species are identified, which is critical for botany and research applications.
  
- **Model Choice**: The **tuned Random Forest model** is recommended for deployment. It offers:
  - **100% recall**
  - **Balanced performance** across secondary metrics
  - **Ability to handle feature interactions** effectively

- **Scalability**: This project can be extended to larger datasets or additional features. The **Random Forest model** is likely to maintain strong performance as the dataset complexity increases.

