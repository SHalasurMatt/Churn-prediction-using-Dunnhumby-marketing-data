# Customer Analytics & CLTV Prediction with Linear Regression

This project demonstrates how to calculate key **customer analytics metrics** such as **Customer Lifetime Value (CLTV), Repeat Rate, Churn Rate, Average Order Value, Purchase Frequency**, and how to build and evaluate a **Linear Regression model** for predictive analytics.

---

## 📊 Key Business Metrics

The following formulas are used:

- **Repeat Rate** =  
  `Number of Customers Who Ordered in Last 3 Months / Total Number of Customers`

- **Churn Rate** =  
  `1 - Repeat Rate`

- **Average Order Value (AOV)** =  
  `Total Revenue / Total Number of Orders`

- **Purchase Frequency (PF)** =  
  `Total Number of Orders / Total Number of Customers`

- **Customer Value (CV)** =  
  `Average Order Value × Purchase Frequency`

- **Customer Lifetime Value (CLTV)** =  
  `((Average Order Value × Purchase Frequency) / Churn Rate) × Profit Margin`

---

## 🤖 Machine Learning

To understand model performance, the dataset is divided into **training** and **test** sets.  

- **Features**: Independent variables (X)  
- **Target**: Dependent variable we want to predict (y)  
- **Test Size**: Split ratio (e.g., 0.2 for 80/20 train–test split)  
- **Random State**: Seed for reproducibility (using `random_state`).  
  - If `None`, a new seed is chosen each run (results vary).  

### Steps:
1. **Import Linear Regression**
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


2. **Split the data**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


3. **Create and train the model**
model = LinearRegression()
model.fit(X_train, y_train)


4. **Make predictions**
y_pred = model.predict(X_test)


5. **Evaluate performance** (MAE, MSE, R², etc.)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


## 🛠️ Project Structure

├── data/ # Raw & processed datasets
├── notebooks/ # Jupyter notebooks for exploration
├── scripts/ # Python scripts for metrics, CLTV, and ML pipeline
│ ├── metrics.py # Functions to calculate CLTV, churn, repeat rate, etc.
│ ├── model.py # Linear Regression implementation
├── requirements.txt # Python dependencies
└── README.md # Project documentation


