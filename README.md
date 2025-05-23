# Car Price Prediction

This project aims to predict the selling price of used cars based on various features like car age, fuel type, transmission, and more, using machine learning.

---

## Project Overview

The price of a car depends on multiple factors such as brand reputation, car features, horsepower, mileage, and others. This project demonstrates how to build a machine learning model to predict car prices using a Random Forest Regressor.

---

## Dataset

- The dataset contains information about used cars, including:
  - Car name
  - Year of manufacture
  - Selling price
  - Present price (manufacturer's price)
  - Driven kilometers
  - Fuel type (Petrol, Diesel, etc.)
  - Selling type (Dealer or Individual)
  - Transmission type (Manual or Automatic)
  - Number of previous owners

---

## Features Engineering

- **Car Age:** Instead of using the manufacturing year directly, the car age is calculated as `current_year - year`.
- Irrelevant columns such as `year` and `Car_Name` are dropped after feature engineering.
- Categorical variables (`Fuel_Type`, `Selling_type`, and `Transmission`) are transformed using one-hot encoding.

---

## Model

- A **Random Forest Regressor** is trained on 80% of the dataset.
- The model is evaluated on the remaining 20% test data.

---

## Evaluation Metrics

- **Root Mean Squared Error (RMSE):** Measures the average prediction error magnitude.
- **R² Score:** Indicates the proportion of variance in the selling price explained by the model.

---

## How to Run

1. Make sure you have Python installed (version 3.7 or higher recommended).
2. Install dependencies:
bash
```
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```
 3. Place the `car data.csv` dataset in the same directory as the script.
 4. Run the script:
bash
```
python car_price_prediction.py
```

5. The script will:
- Load and preprocess the data.
- Train the model.
- Print evaluation metrics.
- Show a plot comparing actual vs predicted selling prices.

---

## Results

The model typically achieves an RMSE around `[0.97]` and an R² score of `[ 0.96]`. The scatter plot visualizes how closely the predicted prices match the actual prices.

---

## Future Improvements

- Add more features like horsepower, torque, mileage, and brand popularity.
- Tune model hyperparameters for better accuracy.
- Experiment with other algorithms like XGBoost or Gradient Boosting.
- Deploy the model using a web app for interactive price prediction.

---

## Author

## Author

**Velagapudi Mahendra** — [LinkedIn Profile](https://www.linkedin.com/in/your-linkedin-profile)


---

## License

This project is open source and available under the MIT License.
