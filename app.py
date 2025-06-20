import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import time
import joblib
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Configure page
st.set_page_config(
    page_title="PayrollGuard AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotting Features
def show_dataset_visuals(data, numeric_cols):
    with st.expander("🔍 Dataset Visual Explorer", expanded=False):
        # Correlation matrix heatmap
        st.markdown("### Feature Correlation Matrix")
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='purples',
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        # Feature distribution comparison
        st.markdown("### Feature Distributions")
        selected_features = st.multiselect(
            "Select up to 4 features to compare:",
            numeric_cols,
            default=numeric_cols[-3]
        )
        if len(selected_features) > 0 :
            for feature in selected_features:
                
                fig = px.histogram(data, x=feature, nbins=30, title=f"Distribution of {feature}")
                fig.update_layout(bargap=0.1)
                fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
    
            
        else:
            st.write("Minimum 1 Features is needed") 

# Custom CSS styling
st.markdown("""
<style>
    /* Main styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 3rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Feature cards */
    .feature-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Warning card */
    .warning-card {
        border-left: 4px solid #f59e0b;
        background-color: #fffbeb;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-item {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s;
    }
    .sidebar-item:hover {
        background-color: #f3f4f6;
    }
    .sidebar-item.active {
        background-color: #eef2ff;
        color: #4f46e5;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.3);
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "landing"

# Sidebar navigation
with st.sidebar:
    st.image('image2.png')
    st.markdown("""
    <div style="padding-bottom:10px; text-align: center;">
        <h1 class="sidebar-title">PAYROLLGUARD</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation options
    
    option = st.selectbox(
        "Select an option:",
        ["🏠 Landing Page", 
         "🤖 Pre-trained Model", 
         "🔧 Custom Model"],
        
    )
    # Model description
    if option == "🤖 Pre-trained Model":
        st.markdown("""
            <div style="padding: 1rem; background-color: #f3f4f6; border-radius: 8px; margin-top: 1rem;">
                <p><strong>Pre-trained Autoencoder</strong></p>
                <p style="font-size: 0.9rem;">Ready-to-use model for payroll anomaly detection with example data.</p>
            </div>
        """, unsafe_allow_html=True)
    
    if option == "🔧 Custom Model":
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ Performance Notice</h4>
            <p style="font-size: 0.9rem;">Custom model training may be slow depending on: </p>
            <ul style="font-size: 0.9rem;">
                <li>Dataset size</li>
                <li>Model complexity</li>
                <li>Hardware capacity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
      
    
# ==================== LANDING PAGE ====================
if option == "🏠 Landing Page":
    st.markdown("""
    <div class="header">
        <h1 style="margin-bottom: 0.5rem;">Payroll Anomaly Detection</h1>
        <p style="margin: 0; font-size: 1.2rem;">AI-powered system for detecting suspicious payroll patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>🚀 About This Project</h2>
        <p>The PayrollGuard AI system leverages deep learning autoencoder technology to identify unusual patterns in payroll data. 
        Our solution helps organizations detect potential errors, fraud, or irregularities in compensation data with advanced anomaly detection algorithms. \n
        1. Automated detection of payroll anomalies
        2. Visual explanation of suspicious patterns
        3. Customizable detection thresholds
        4. Detailed reporting for investigation
                
        
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠 Available Models")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Pre-trained Model</h3>
            <p>Get started immediately with our optimized autoencoder trained on payroll patterns:</p>
            <ul>
                <li>Quick one-click analysis</li>
                <li>Example dataset included</li>
                <li>Optimized for performance</li>
                <li>Ready for production use</li>
            </ul>
            
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Custom Model</h3>
            <p>Bring your own model and data for customized detection:</p>
            <ul>
                <li>Upload your trained autoencoder</li>
                <li>Use your organization's payroll data</li>
                <li>Adjust detection parameters</li>
                <li>Feature importance analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
       
    
    
    

# ==================== PRETRAINED MODEL PAGE ====================
if option == "🤖 Pre-trained Model":
    st.session_state.option = "🤖 Pre-trained Model"
# Streamlit page config
    
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

        # Expander
    st.markdown("### Payroll Data Overview")
    st.dataframe(
        data.tail(10).style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

    with st.expander("⚙️ Detection Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Anomaly Threshold", 
                    0.1, 10.0, 1.5, 0.1,
                    help="Adjust the sensitivity for anomaly detection"
                )
                
            
            with col2:
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found in the data!")
                else:
                    features = st.multiselect(
                        "Select Features for Analysis",
                        numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))],
                        help="Choose which numeric features to include in the analysis"
                    )
    # Main preview

    
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

        show_dataset_visuals(data , numeric_cols)





# ==================== CUSTOM MODEL PAGE ====================
elif option == "🔧 Custom Model":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <h1>Custom Model Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-card">
        <h4>⚠️ Performance Considerations</h4>
        <p>Using custom models may require additional processing time:</p>
        <ul>
            <li>Model loading and initialization</li>
            <li>Data preprocessing</li>
            <li>Feature scaling and transformation</li>
            <li>Anomaly scoring</li>
        </ul>
        <p><strong>Expected time:</strong> 30 seconds - 5 minutes depending on dataset size</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploaders
    st.markdown("## Upload Your Components")
        
    with st.expander("Data Upload", expanded=True):
        data_file = st.file_uploader(
            "Select Payroll Data File",
            type=['csv', 'xlsx'],
            help="Upload your payroll data (CSV or Excel format)"
        )
    
    # Configuration section
            
    if data_file:
        try:
            @st.cache_data(show_spinner=True)
            def loading():
            # Load model and data
                model = joblib.load('autoencoder_model.pkl')


                if data_file.name.endswith('.csv'):
                    data = pd.read_csv(data_file)
                else:
                    data = pd.read_excel(data_file)

                st.success("Model and data loaded successfully!")
                return model , data
            model , data = loading()
            
            # Analysis configuration
            with st.expander("⚙️ Detection Parameters", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider(
                        "Anomaly Threshold", 
                        0.1, 10.0, 1.5, 0.1,
                        help="Set the MSE threshold for anomaly classification"
                    )
                    min_mse = st.number_input(
                        "Minimum MSE to Highlight", 
                        0.0, 20.0, 2.0, 0.1
                    )
                
                with col2:
                    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                    if not numeric_cols:
                        st.warning("No numeric columns found in the data!")
                    else:
                        features = st.multiselect(
                            "Select Features for Analysis",
                            numeric_cols,
                            default=numeric_cols[:min(10, len(numeric_cols))],
                            help="Choose which numeric features to include in the analysis"
                        )
            
            # Run analysis button
            if st.button("Run Custom Analysis", type="primary"):
                with st.spinner("Processing custom data... This may take some time"):
                    start_time = time.time()
                    
                    try:
                        # Data preprocessing
                        scaler = StandardScaler()
                        X = data[features].values
                        X_scaled = scaler.fit_transform(X)
                        
                        # Get predictions
                        reconstructions = model.predict(X_scaled)
                        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                        
                        # Create results
                        results = data.copy()
                        results['MSE'] = mse
                        results['Anomaly'] = mse > threshold
                        results['Anomaly_Score'] = (mse - threshold) / threshold
                        
                        # Calculate feature importance
                        feature_errors = np.mean(np.abs(X_scaled - reconstructions), axis=0)
                        feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Contribution_to_Error': feature_errors
                        }).sort_values('Contribution_to_Error', ascending=False)
                        
                        # Performance metrics
                        duration = time.time() - start_time
                        anomaly_count = results['Anomaly'].sum()
                        anomaly_pct = (anomaly_count / len(results)) * 100
                        
                        # Store results
                        st.session_state.custom_results = results
                        st.session_state.custom_feature_importance = feature_importance
                        st.session_state.custom_metrics = {
                            'duration': duration,
                            'anomaly_count': anomaly_count,
                            'anomaly_pct': anomaly_pct,
                            'threshold': threshold
                        }
                        
                        st.success(f"Analysis completed in {duration:.2f} seconds")
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            
            # Display results if available
            if 'custom_results' in st.session_state:
                results = st.session_state.custom_results
                metrics = st.session_state.custom_metrics
                feature_importance = st.session_state.custom_feature_importance
                
                st.markdown("## 📊 Custom Analysis Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", len(results))
                col2.metric("Anomalies Detected", f"{metrics['anomaly_count']} ({metrics['anomaly_pct']:.1f}%)")
                col3.metric("Processing Time", f"{metrics['duration']:.2f} sec")
                
                tab1, tab2 = st.tabs(["📈 Error Analysis", "📌 Feature Importance"])
                
                with tab1:
                    st.markdown("### Error Distribution")
                    fig = px.histogram(
                        results,
                        x='MSE',
                        nbins=50,
                        labels={'MSE': 'Mean Squared Error'},
                        color_discrete_sequence=['#4f46e5']
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Top Anomalies")
                    st.dataframe(
                        results[results['Anomaly']]
                        .sort_values('MSE', ascending=False)
                        .head(50)
                        .style.background_gradient(subset=['MSE'], cmap='Reds'),
                        use_container_width=True,
                        height=500
                    )
                
                with tab2:
                    st.markdown("### Feature Importance")
                    fig2 = px.bar(
                        feature_importance,
                        x='Contribution_to_Error',
                        y='Feature',
                        orientation='h',
                        color='Contribution_to_Error',
                        color_continuous_scale='Bluered'
                    )
                    fig2.update_layout(height=600)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Download buttons
                st.markdown("### Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name="custom_payroll_anomalies.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    buf = BytesIO()
                    joblib.dump({
                        'results': results,
                        'feature_importances': feature_importance,
                        'metrics': metrics
                    }, buf)
                    buf.seek(0)
                    st.download_button(
                        label="Download Full Analysis (Pickle)",
                        data=buf,
                        file_name="payroll_analysis_results.pkl",
                        mime="application/octet-stream"
                    )
                st.subheader("Data Analysis")
                            
                show_dataset_visuals(data , numeric_cols)
               



        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
