import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import datetime
import os

# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="Crime Shield AI | Pro",
    page_icon="🛡️",
    layout="wide"
)

# --- ADVANCED STYLING ---
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- FILE PATHS ---
DATA_PATH = "data/crime_dataset_india.csv"
MODEL_PATH = "models/crime_prediction_model.pkl"
CITY_ENCODER_PATH = "models/le_city.pkl"
DOMAIN_ENCODER_PATH = "models/le_domain.pkl"
METRICS_PATH = "outputs/model_comparison.csv"
MODEL_PLOT_PATH = "outputs/plots/model_comparison.png"
FEATURE_PLOT_PATH = "outputs/plots/feature_importance.png"

# --- LOAD MODEL AND ENCODERS ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le_city = joblib.load(CITY_ENCODER_PATH)
    le_domain = joblib.load(DOMAIN_ENCODER_PATH)
    return model, le_city, le_domain

# --- LOAD DATA ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df['Date of Occurrence'] = pd.to_datetime(
        df['Date of Occurrence'],
        format='mixed',
        dayfirst=True
    )
    df['Month'] = df['Date of Occurrence'].dt.month
    df['Day'] = df['Date of Occurrence'].dt.day
    df['DayOfWeek'] = df['Date of Occurrence'].dt.dayofweek
    df['Year'] = df['Date of Occurrence'].dt.year

    df['Hour'] = pd.to_datetime(
        df['Time of Occurrence'],
        format='mixed'
    ).dt.hour

    df['Weapon Used'] = df['Weapon Used'].fillna('Not Specified')

    return df


# --- LOAD RESOURCES ---
try:
    df = load_and_clean_data()
    model, le_city, le_domain = load_model()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/100/shield.png", width=80)
st.sidebar.title("CRIME SHIELD AI")
st.sidebar.markdown("*Predictive Policing & Analysis System*")
st.sidebar.divider()

menu = st.sidebar.selectbox(
    "COMMAND CENTER",
    [
        "Strategic Dashboard",
        "AI Risk Intelligence",
        "City Safety Rankings",
        "Model Performance"
    ]
)

# --- PAGE 1: STRATEGIC DASHBOARD ---
if menu == "Strategic Dashboard":
    st.title("📊 Strategic Crime Analytics")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Incidents", f"{len(df):,}")
    m2.metric("Operational Cities", df['City'].nunique())
    m3.metric("Closure Rate", f"{(df['Case Closed'] == 'Yes').mean():.1%}")
    m4.metric("Avg Police Presence", f"{df['Police Deployed'].mean():.1f} Units")

    st.divider()

    c1, c2 = st.columns([6, 4])

    with c1:
        st.subheader("📈 Temporal Crime Density (Hourly)")
        hourly_data = df.groupby('Hour').size().reset_index(name='Incidents')
        fig_hour = px.area(
            hourly_data,
            x='Hour',
            y='Incidents',
            color_discrete_sequence=['#ff4b4b'],
            template="plotly_dark"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        st.subheader("⚔️ Weapon Distribution")
        weapon_data = df['Weapon Used'].value_counts().nlargest(5)
        fig_weapon = px.pie(
            values=weapon_data.values,
            names=weapon_data.index,
            hole=0.6,
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_weapon, use_container_width=True)

    st.subheader("🗺️ Crime Domain Breakdown")
    fig_sun = px.sunburst(
        df,
        path=['Crime Domain', 'Crime Description'],
        color='Crime Domain',
        template="plotly_dark"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

# --- PAGE 2: AI RISK INTELLIGENCE ---
elif menu == "AI Risk Intelligence":
    st.title("🔮 Predictive Intelligence Engine")
    st.info("This prediction is generated using a trained machine learning model.")

    with st.expander("Configure Prediction Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        sel_city = col1.selectbox(
            "Target Region",
            sorted(df['City'].unique())
        )

        sel_date = col2.date_input(
            "Prediction Date",
            datetime.date.today()
        )

        sel_domain = col3.selectbox(
            "Crime Category",
            sorted(df['Crime Domain'].unique())
        )

    if st.button("RUN AI SIMULATION"):
        try:
            city_e = le_city.transform([sel_city])[0]
            domain_e = le_domain.transform([sel_domain])[0]
            month = sel_date.month
            dow = sel_date.weekday()

            prediction = model.predict(
                [[city_e, month, dow, domain_e]]
            )[0]

            risk_val = min(100, int(prediction * 10))

            st.divider()
            res1, res2 = st.columns([1, 2])

            with res1:
                st.markdown(f"### Risk Index: {risk_val}/100")
                if risk_val > 70:
                    st.error("🚨 CRITICAL ALERT: High probability of incident.")
                elif risk_val > 35:
                    st.warning("⚠️ CAUTION: Elevated risk detected.")
                else:
                    st.success("✅ STABLE: Normal safety levels.")

            with res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_val,
                    title={'text': "Security Threat Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b"},
                        'steps': [
                            {'range': [0, 40], 'color': "green"},
                            {'range': [40, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- PAGE 3: CITY SAFETY RANKINGS ---
elif menu == "City Safety Rankings":
    st.title("🏆 Regional Safety Index")
    st.markdown("Ranking based on Incident Volume vs Police Deployment efficiency.")

    rank_df = df.groupby('City').agg({
        'Report Number': 'count',
        'Police Deployed': 'mean'
    }).reset_index()

    rank_df['Safety_Score'] = (
        rank_df['Police Deployed'] /
        rank_df['Report Number']
    ) * 100

    rank_df = rank_df.sort_values(by='Safety_Score', ascending=False)

    st.dataframe(rank_df.rename(columns={
        'Report Number': 'Total Crimes',
        'Police Deployed': 'Avg Police Strength',
        'Safety_Score': 'Safety Index'
    }), use_container_width=True)

    fig_rank = px.bar(
        rank_df,
        x='City',
        y='Safety_Score',
        color='Safety_Score',
        color_continuous_scale='RdYlGn',
        title="City Safety Ranking (Higher is Safer)"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

# --- PAGE 4: MODEL PERFORMANCE ---
elif menu == "Model Performance":
    st.title("📊 Model Performance & Research Metrics")

    st.subheader("📈 Model Comparison Metrics")
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Run train_model.py to generate metrics.")

    st.subheader("📊 Model Comparison Plot")
    if os.path.exists(MODEL_PLOT_PATH):
        st.image(MODEL_PLOT_PATH, use_container_width=True)
    else:
        st.warning("Model comparison plot not found.")

    st.subheader("🔍 Feature Importance")
    if os.path.exists(FEATURE_PLOT_PATH):
        st.image(FEATURE_PLOT_PATH, use_container_width=True)
    else:
        st.warning("Feature importance plot not found.")