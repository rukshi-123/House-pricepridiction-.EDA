erytinsdvn## House-pricepridiction-EDA
# 🏠 House Price Prediction Project

Welcome to the **House Price Prediction** project! This repository showcases how to use machine learning models to predict house prices based on various features of a house. The analysis leverages Python's powerful data-science libraries and several machine learning algorithms.

---

## 📁 Dataset Overview

The dataset (`house.csv`) has **2,000 rows** and these key features:

| Feature      | Description                         |
|--------------|-------------------------------------|
| Id           | Unique house ID                     |
| Area         | Floor area (sq. ft.)                |
| Bedrooms     | Number of bedrooms                  |
| Bathrooms    | Number of bathrooms                 |
| Floors       | Number of floors                    |
| YearBuilt    | Year the house was built            |
| Location     | Location category (e.g., Downtown)  |
| Condition    | House condition (e.g., Excellent)   |
| Garage       | Garage availability (Yes/No)        |
| Price        | Target variable (house price 💰)     |

---

## 🛠️ Libraries Used

Here’s a rundown of the key Python libraries used in the notebook:

| Library           | Purpose                                                      |
|-------------------|-------------------------------------------------------------|
| `numpy`           | Numerical operations (arrays & math)                        |
| `pandas`          | Data manipulation and analysis (DataFrames)                  |
| `matplotlib`      | Visualization (plots and charts)                            |
| `seaborn`         | Statistical data visualization                              |
| `sklearn`         | Machine learning models, preprocessing, and evaluation       |
| `warnings`        | Suppressing unnecessary output                               |

---

## 🔍 Data Preprocessing

- **Reading Data**: The CSV file is read into a DataFrame.
- **Exploration**: Used methods like `.head()` and `.describe()` for a first look at the data.
- **Encoding Categorical Data**: Used `LabelEncoder` to convert categorical columns (`Location`, `Condition`, `Garage`) into numerical form, which is required for ML models.

---

## 📊 Exploratory Data Analysis

Visualizations are created to understand the distribution of data and relationships between features:

- 📌 **Correlation heatmaps** to study feature interdependence.
- 📊 **Distribution plots** for variables like `Area` and `Price`.

---

## 🤖 Machine Learning Models Applied

Several ML models were applied to predict house prices:

| Model                   | 📌 Icon | Description                                                                        |
|-------------------------|--------|------------------------------------------------------------------------------------|
| Linear Regression       | 📈     | Baseline regression model for continuous targets                                   |
| Lasso Regression        | 🦾     | Linear regression with L1 regularization (feature selection & shrinkage)           |
| Ridge Regression        | 🧊     | Linear regression with L2 regularization (penalizes large coefficients)            |
| Random Forest Regressor | 🌳     | Ensemble method using multiple decision trees for better accuracy & robustness     |

Each model is trained on a split of the data using `train_test_split`, and evaluated using the **R² score**.

---

## 🚦 Project Workflow

1. **Import Libraries**  
2. **Load & Explore Data**  
3. **Data Cleaning & Encoding**  
4. **Feature Selection & Splitting**  
5. **Model Training**  
6. **Prediction & Evaluation**

---

## 📝 Example Code Snippet

```
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# Model initialization
lasso = Lasso()
ridge = Ridge()
rf = RandomForestRegressor(n_estimators=10)

# Model training
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_rf = rf.predict(X_test)
```

---

## 📈 Visual Summary

📌 **Correlation Heatmap**  
Analyze relationships between variables to select important predictors.

📌 **Price Distribution**  
Visualize the spread of house prices and identify outliers.

---

## ✨ Conclusion

This notebook demonstrates:

- ✅ End-to-end workflow for ML regression projects.
- ✅ How to use, compare, and interpret various regression models.
- ✅ The power of data encoding and visualization in driving insight & model accuracy.

---

## 🚀 Quickstart

1. Install requirements:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Launch the notebook and run through the workflow!

---

### ⭐ Happy Predicting! 🏡📊✨
```

✔️ **How to use:**

1. Create a file in your GitHub repo root directory called `README.md`.
2. Paste the above Markdown code inside that file.
3. Commit the file to your repo.
4. GitHub will render the README beautifully with emojis, tables, code blocks, and structure!



