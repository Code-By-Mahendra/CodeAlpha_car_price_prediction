import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('car data.csv')
print("✅ Dataset loaded successfully!\n")

# Show first 5 rows
print(" First 5 rows:")
print(df.head())

# Dataset info
print("\n Data Info:\n")
print(df.info())

# Describe dataset
print("\ Summary statistics:")
print(df.describe())

# Check missing values
print("\n Missing values:")
print(df.isnull().sum())

# Feature engineering: Create car age
current_year = 2025
df['car_age'] = current_year - df['year']

# Drop irrelevant columns
df.drop(['year', 'Car_Name'], axis=1, inplace=True)

# One-Hot Encoding for categorical features
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), categorical_features)],
                       remainder='passthrough')

X = ct.fit_transform(df.drop('Selling_Price', axis=1))
y = df['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.grid(True)
plt.show()
