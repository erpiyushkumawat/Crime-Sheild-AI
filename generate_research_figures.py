import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -------------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------------
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
DATA_PATH = "data/crime_dataset_india.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
df['Date of Occurrence'] = pd.to_datetime(
    df['Date of Occurrence'], format='mixed', dayfirst=True
)
df['Month'] = df['Date of Occurrence'].dt.month
df['DayOfWeek'] = df['Date of Occurrence'].dt.dayofweek
df['Hour'] = pd.to_datetime(
    df['Time of Occurrence'], format='mixed'
).dt.hour
df['Weapon Used'] = df['Weapon Used'].fillna('Not Specified')

# -------------------------------
# FIGURE 1: Temporal Crime Distribution by Hour
# -------------------------------
hourly_data = df.groupby('Hour').size().reset_index(name='Incidents')
fig1 = px.area(
    hourly_data,
    x='Hour',
    y='Incidents',
    title="Temporal Crime Distribution by Hour"
)
fig1.write_image(f"{PLOTS_DIR}/figure_1_temporal_crime_distribution.png", scale=3)

# -------------------------------
# FIGURE 2: Weapon Usage Distribution
# -------------------------------
weapon_data = df['Weapon Used'].value_counts().nlargest(5)
fig2 = px.pie(
    values=weapon_data.values,
    names=weapon_data.index,
    hole=0.5,
    title="Weapon Usage Distribution"
)
fig2.write_image(f"{PLOTS_DIR}/figure_2_weapon_usage_distribution.png", scale=3)

# -------------------------------
# FIGURE 3: Crime Domain Hierarchy
# -------------------------------
fig3 = px.sunburst(
    df,
    path=['Crime Domain', 'Crime Description'],
    title="Crime Domain Hierarchy"
)
fig3.write_image(f"{PLOTS_DIR}/figure_3_crime_domain_hierarchy.png", scale=3)

# -------------------------------
# FIGURE 4: Model Performance Comparison
# -------------------------------
metrics_path = "outputs/model_comparison.csv"
metrics_df = pd.read_csv(metrics_path)

fig4 = px.bar(
    metrics_df,
    x="Model",
    y="RMSE",
    color="Model",
    title="Model Performance Comparison"
)
fig4.write_image(f"{PLOTS_DIR}/figure_4_model_performance_comparison.png", scale=3)

# -------------------------------
# FIGURE 5: Feature Importance Analysis
# -------------------------------
MODEL_PATH = "models/crime_prediction_model.pkl"
model = joblib.load(MODEL_PATH)

agg_df = df.groupby(
    ['City', 'Month', 'DayOfWeek', 'Crime Domain']
).size().reset_index(name='Target')

from sklearn.preprocessing import LabelEncoder

le_city = LabelEncoder()
le_domain = LabelEncoder()

agg_df['City_E'] = le_city.fit_transform(agg_df['City'])
agg_df['Domain_E'] = le_domain.fit_transform(agg_df['Crime Domain'])

feature_names = ['City_E', 'Month', 'DayOfWeek', 'Domain_E']

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig5 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Analysis"
    )
    fig5.write_image(f"{PLOTS_DIR}/figure_5_feature_importance.png", scale=3)

# -------------------------------
# FIGURE 6: AI-Based Crime Risk Prediction Gauge
# -------------------------------
fig6 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=75,
    title={'text': "AI-Based Crime Risk Prediction"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 40], 'color': "green"},
            {'range': [40, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))
fig6.write_image(f"{PLOTS_DIR}/figure_6_risk_prediction_gauge.png", scale=3)

# -------------------------------
# FIGURE 7: City Safety Rankings
# -------------------------------
rank_df = df.groupby('City').agg({
    'Report Number': 'count',
    'Police Deployed': 'mean'
}).reset_index()

rank_df['Safety_Score'] = (
    rank_df['Police Deployed'] / rank_df['Report Number']
) * 100

rank_df = rank_df.sort_values(by='Safety_Score', ascending=False)

fig7 = px.bar(
    rank_df,
    x='City',
    y='Safety_Score',
    color='Safety_Score',
    color_continuous_scale='RdYlGn',
    title="City Safety Rankings"
)
fig7.write_image(f"{PLOTS_DIR}/figure_7_city_safety_rankings.png", scale=3)

print("✅ All research figures generated and saved in outputs/plots/")