import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/crime_dataset_india.csv")
print(df.head())

# Feature Engineering
df['Date of Occurrence'] = pd.to_datetime(
    df['Date of Occurrence'], format='mixed', dayfirst=True
)
df['Month'] = df['Date of Occurrence'].dt.month
df['DayOfWeek'] = df['Date of Occurrence'].dt.dayofweek

# Aggregate data
agg_df = df.groupby(
    ['City', 'Month', 'DayOfWeek', 'Crime Domain']
).size().reset_index(name='Target')

# Encode categorical variables
le_city = LabelEncoder()
le_domain = LabelEncoder()

agg_df['City_E'] = le_city.fit_transform(agg_df['City'])
agg_df['Domain_E'] = le_domain.fit_transform(agg_df['Crime Domain'])

X = agg_df[['City_E', 'Month', 'DayOfWeek', 'Domain_E']]
y = agg_df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models for comparison
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_model = None
best_r2 = -np.inf
best_name = ""

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nModel Comparison:\n", results_df)
print(f"\nBest Model: {best_name}")

# Plot Model Comparison
results_df.set_index("Model")[["RMSE"]].plot(kind="bar", legend=False)
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("outputs/plots/model_comparison.png")
plt.close()

# Feature Importance (if available)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = X.columns

    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()

# Save the model and encoders
joblib.dump(best_model, "models/crime_prediction_model.pkl")
joblib.dump(le_city, "models/le_city.pkl")
joblib.dump(le_domain, "models/le_domain.pkl")

print("✅ Model and encoders saved successfully!")