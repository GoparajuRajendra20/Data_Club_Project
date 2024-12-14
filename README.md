# CDC Diabetes Prediction Based on Health Indicators

## Overview
This project focuses on predicting diabetes (healthy, pre-diabetic, or diabetic) based on health indicators using machine learning models. The dataset, **CDC Diabetes Health Indicators**, was sourced from the UC Irvine Machine Learning Repository, consisting of **253,680 instances**. The goal is to clean, preprocess, and analyze the data, implement various models, and identify the best-performing model for classification.

## Dataset
- **Source**: CDC-funded dataset from UC Irvine Machine Learning Repository.
- **Size**: 253,680 records.
- **Features**: Health indicators such as BMI, General Health, Mental Health, Physical Health, Age, Education, and Income.
- **Quality**: No missing values and real-world health data.

## Methodology
### 1. Data Cleaning and Preprocessing
- Changed variable data types to integers.
- Removed outliers in the `BMI` column using the **Interquartile Range (IQR)** method.
- Dropped duplicate rows and standardized non-binary features.
- Shuffled data to eliminate biases during training.

### 2. Exploratory Data Analysis (EDA)
- Created a correlation heatmap to analyze feature relationships.
- Kept all features for model development as correlations between variables were weak (< 0.51).

### 3. Data Division
- **Train-Test Split**: 70% training, 30% testing.
- **Validation Split**: 50% of the test data for validation.
- Two iterations were performed:
  - Using all features.
  - Using selected features after feature selection.

### 4. Model Selection
Implemented the following models:
- **Random Forest Classifier** (SciKit-Learn)
- **Neural Networks Classifier** (Keras)
- **XGBoost Classifier**

The **Decision Tree Classifier** was dropped due to inferior performance compared to the Random Forest.

### 5. Feature Selection
Used **ExtraTreesClassifier** with two approaches:
1. Tree-based feature importance.
2. Pipeline-based feature importance using Mean Decrease in Impurity (MDI).

Top selected features:
- `BMI_standardized`
- `GenHlth_standardized`
- `MentHlth_standardized`
- `PhysHlth_standardized`
- `Age_standardized`
- `Education_standardized`
- `Income_standardized`

### 6. Hyperparameter Tuning
Performed **GridSearchCV** to optimize the models:
- **Random Forest Classifier**: Optimized `n_estimators`, `max_depth`, and `min_samples_split`.
- **Neural Networks Classifier**: Tuned `units`, `learning_rate`, `batch_size`, `epochs`, `optimizer`, and `activation`.
- **XGBoost Classifier**: Adjusted `learning_rate`, `n_estimators`, `max_depth`, and `gamma`.

## Results
The following models were evaluated:
1. **Random Forest Classifier**
2. **Neural Networks Classifier**
3. **XGBoost Classifier**

### Best Model
- **Keras Neural Networks Classifier** with optimal hyperparameters outperformed others in predicting diabetes status.
- Metrics indicate strong performance across all evaluation datasets.

## Future Work
- Use **GPU acceleration** for faster hyperparameter tuning and model training.
- Develop **custom models** to compare with SciKit-Learn, Keras, and XGBoost.
- Explore **manual hyperparameter tuning** for further optimization.

## Contributors
- **Luke Abbatessa**
- **Rajendra Goparaju**

## Tools and Technologies
- **Libraries**: SciKit-Learn, Keras, XGBoost, Pandas, NumPy.
- **Techniques**: Hyperparameter tuning (GridSearchCV), Feature Selection (Tree-based methods), Data Preprocessing.
- **Languages**: Python.

---
**Project Goal**: To predict diabetes based on health indicators and identify the most important features influencing the outcomes.
