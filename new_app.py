import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import time
import joblib

# Streamlit page config
st.set_page_config(
    page_title="PayrollGuard Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting custom CSS
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .card, .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .metric-card { text-align: center; }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f46e5;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 style="margin:0;font-size:2rem;">PayrollGuard Pro</h1>
    <p style="margin:0;opacity:0.8;">Advanced Anomaly Detection for Payroll Systems</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(show_spinner=True)
def load_data():
    return pd.read_excel("Payroll Data.xlsx")

@st.cache_resource(show_spinner=True)
def load_model():
    return joblib.load("autoencoder_model.pkl")

data = load_data()
model = load_model()

# Sidebar config
with st.sidebar:
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)

    threshold = st.slider(
        "Anomaly Threshold", 0.5, 5.0, 1.5, 0.1,
        help="Adjust the sensitivity for anomaly detection"
    )

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect(
        "Select Features to Analyze", numeric_cols,
        default=numeric_cols,
        help="Choose which numerical features to include in the analysis"
    )

    st.markdown("---")

# Main preview

st.markdown("### Payroll Data Overview")
st.dataframe(
    data.tail(10).style.background_gradient(subset=numeric_cols, cmap='Blues'),
    use_container_width=True
)
st.markdown("### Quick Analysis")
if st.button("🚀 Run Detection", type="primary"):
    start_time = time.time()
    with st.spinner("Detecting anomalies..."):
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reconstructions = model.predict(X_scaled)
        mse = np.mean((X_scaled - reconstructions) ** 2, axis=1)
        results = data.copy()
        results['MSE'] = mse
        results['Anomaly'] = mse > threshold
        results['Anomaly_Score'] = (mse - threshold) / threshold
        duration = time.time() - start_time
        anomaly_count = int(results['Anomaly'].sum())
        anomaly_pct = (anomaly_count / len(results)) * 100
        st.session_state.results = results
        st.session_state.metrics = {
            'duration': duration,
            'anomaly_count': anomaly_count,
            'anomaly_pct': anomaly_pct
        }
        st.success("Analysis completed successfully!")

# If results available
if 'results' in st.session_state:
    results = st.session_state.results
    metrics = st.session_state.metrics

    st.markdown("## Detection Results")

    cols = st.columns(4)
    cols[0].markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Processed</div>
        <div class="metric-value">{len(results):,}</div>
        <div class="metric-label">records</div>
    </div>""", unsafe_allow_html=True)

    cols[1].markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Anomalies</div>
        <div class="metric-value">{metrics['anomaly_count']}</div>
        <div class="metric-label">found</div>
    </div>""", unsafe_allow_html=True)

    cols[2].markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rate</div>
        <div class="metric-value">{metrics['anomaly_pct']:.1f}%</div>
        <div class="metric-label">of total</div>
    </div>""", unsafe_allow_html=True)

    cols[3].markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Duration</div>
        <div class="metric-value">{metrics['duration']:.2f}s</div>
        <div class="metric-label">processing</div>
    </div>""", unsafe_allow_html=True)

    # Visual tabs
    tab1, tab2 = st.tabs(["📊 Error Distribution", "🔍 Top Anomalies"])

    with tab1:
        fig = px.histogram(results, x='MSE', nbins=50, color_discrete_sequence=['#4f46e5'])
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text=f"Threshold: {threshold}", annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Detailed Anomalies View")
        anomaly_df = results[results['Anomaly']].sort_values(by='MSE', ascending=False)

        if 'department' in anomaly_df.columns:
            dept_counts = anomaly_df['department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**By Department**")
                st.dataframe(
                    dept_counts.style.background_gradient(subset=['Count'], cmap='Reds'),
                    use_container_width=True, height=300
                )

        st.markdown("**Anomaly Details**")
        st.dataframe(
            anomaly_df.head(20).style.background_gradient(subset=['MSE'], cmap='Reds'),
            use_container_width=True, height=400
        )

    # Feature importance
    import matplotlib.pyplot as plt
    feature_importance = np.mean(np.abs(results[features] - results[features].mean()), axis=0)
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance, color='skyblue')
    plt.xlabel('Mean Absolute Error')
    plt.title('Feature Importance based on Reconstruction Error')
    plt.grid(axis='x')
    st.pyplot(plt)
