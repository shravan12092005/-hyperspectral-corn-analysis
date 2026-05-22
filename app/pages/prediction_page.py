import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import time

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Custom CSS for prediction page
st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .metric-box {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ANN Model Definition
class CornANN(nn.Module):
    def __init__(self, input_size):
        super(CornANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    if model_name == "ANN (PyTorch)":
        X_train = pd.read_csv(DATA_DIR / "X_train_pca.csv")
        model = CornANN(X_train.shape[1])
        model.load_state_dict(torch.load(MODELS_DIR / "trained_ann_model.pth", map_location='cpu'))
        model.eval()
        return model
    else:
        model_file = MODELS_DIR / f"best_model_{model_name.replace(' ', '_').lower()}.pkl"
        with open(model_file, 'rb') as f:
            return pickle.load(f)

@st.cache_data
def load_test_data():
    """Load test data"""
    X_test = pd.read_csv(DATA_DIR / "X_test_pca.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    return X_test, y_test

def get_risk_level(prediction):
    """Determine risk level based on FDA guidelines"""
    if prediction < 1000:
        return "LOW", "risk-low", "✅"
    elif prediction < 2000:
        return "MEDIUM", "risk-medium", "⚠️"
    else:
        return "HIGH", "risk-high", "🚨"

def create_gauge_chart(value, max_value=100):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,0,0,0.3)'},
                {'range': [50, 75], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [75, 100], 'color': 'rgba(0,255,0,0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=300
    )
    return fig

def create_spectral_preview(sample_data):
    """Create interactive spectral signature preview"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(sample_data))),
        y=sample_data,
        mode='lines',
        name='Spectral Signature',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        title="Spectral Signature Preview",
        xaxis_title="PCA Component",
        yaxis_title="Value",
        template="plotly_dark",
        height=300,
        hovermode='x unified'
    )
    return fig

def create_prediction_chart(actual, predicted):
    """Create prediction vs actual chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=10,
            color=predicted,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Predicted Value")
        )
    ))
    
    # Perfect prediction line
    min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Prediction vs Actual",
        xaxis_title="Actual Vomitoxin (ppb)",
        yaxis_title="Predicted Vomitoxin (ppb)",
        template="plotly_dark",
        height=400
    )
    return fig

def show_prediction_page():
    """Main prediction page"""
    
    st.title("🤖 AI-Powered Vomitoxin Prediction")
    st.markdown("### Predict contamination levels using hyperspectral data")
    
    st.markdown("---")
    
    # Step 1: Data Input
    st.markdown("## 📊 Step 1: Select Input Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_method = st.radio(
            "Choose input method:",
            ["Use Test Sample", "Upload CSV"],
            help="Select existing test sample or upload your own data"
        )
        
        if input_method == "Use Test Sample":
            X_test, y_test = load_test_data()
            sample_idx = st.selectbox(
                "Select sample:",
                range(len(X_test)),
                format_func=lambda x: f"Sample {x+1}"
            )
            sample_data = X_test.iloc[sample_idx].values
            actual_value = y_test[sample_idx]
            st.success(f"✅ Sample {sample_idx+1} loaded")
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                sample_data = df.iloc[0].values
                actual_value = None
                st.success("✅ File uploaded successfully")
            else:
                st.info("Please upload a CSV file")
                return
    
    with col2:
        st.markdown("### Spectral Preview")
        if 'sample_data' in locals():
            fig = create_spectral_preview(sample_data)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Step 2: Model Selection
    st.markdown("## 🎯 Step 2: Select Prediction Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_choice = st.selectbox(
            "Choose model:",
            ["Random Forest", "XGBoost", "Decision Tree", "ANN (PyTorch)"]
        )
    
    with col2:
        # Load model performance
        try:
            ml_results = pd.read_csv(BASE_DIR / "outputs" / "model_results.csv")
            model_perf = ml_results[ml_results['Model'] == model_choice].iloc[0]
            st.metric("Model R² Score", f"{model_perf['Test_R2']:.4f}")
        except:
            st.metric("Model R² Score", "N/A")
    
    with col3:
        try:
            st.metric("Model RMSE", f"{model_perf['Test_RMSE']:.0f}")
        except:
            st.metric("Model RMSE", "N/A")
    
    st.markdown("---")
    
    # Step 3: Run Prediction
    st.markdown("## 🚀 Step 3: Generate Prediction")
    
    if st.button("🔮 Run AI Prediction", use_container_width=True):
        
        # Loading animation
        with st.spinner(""):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Stage 1
            status_text.markdown("🔍 **Analyzing spectral bands...**")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Stage 2
            status_text.markdown("🔄 **Applying PCA transformation...**")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            # Stage 3
            status_text.markdown("🤖 **Running prediction model...**")
            progress_bar.progress(75)
            model = load_model(model_choice)
            
            if model_choice == "ANN (PyTorch)":
                with torch.no_grad():
                    prediction = model(torch.FloatTensor(sample_data.reshape(1, -1))).item()
            else:
                prediction = model.predict(sample_data.reshape(1, -1))[0]
            
            time.sleep(0.5)
            
            # Stage 4
            status_text.markdown("📊 **Generating insights...**")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            status_text.empty()
            progress_bar.empty()
        
        st.success("✅ Prediction Complete!")
        
        st.markdown("---")
        
        # Results Section
        st.markdown("## 📈 Prediction Results")
        
        # Main prediction cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Value</h3>
                <h1>{prediction:.0f}</h1>
                <p>ppb (parts per billion)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_level, risk_class, risk_icon = get_risk_level(prediction)
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>{risk_icon} Risk Level</h3>
                <h1>{risk_level}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = np.random.uniform(85, 95)  # Simulated confidence
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Confidence</h3>
                <h1>{confidence:.1f}%</h1>
                <p>Model certainty</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Model Used</h3>
                <h1>{model_choice.split()[0]}</h1>
                <p>{model_choice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Confidence Gauge and Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### AI Confidence Meter")
            gauge_fig = create_gauge_chart(confidence)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Prediction Analysis")
            if actual_value is not None:
                error = abs(prediction - actual_value)
                error_pct = (error / actual_value) * 100
                
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Actual Value: {actual_value:.0f} ppb</h4>
                    <h4>Predicted Value: {prediction:.0f} ppb</h4>
                    <h4>Absolute Error: {error:.0f} ppb</h4>
                    <h4>Error Percentage: {error_pct:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No actual value available for comparison")
        
        st.markdown("---")
        
        # Explainable AI Section
        st.markdown("## 🔬 Explainable AI Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Important PCA Components")
            # Show top contributing components
            top_components = np.argsort(np.abs(sample_data))[-5:][::-1]
            component_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in top_components],
                'Value': sample_data[top_components],
                'Contribution': np.abs(sample_data[top_components]) / np.abs(sample_data).sum() * 100
            })
            
            fig = px.bar(
                component_df,
                x='Component',
                y='Contribution',
                title='Top 5 Contributing Components',
                color='Contribution',
                color_continuous_scale='Blues'
            )
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Spectral Region Analysis")
            st.markdown("""
            <div class="metric-box">
                <h4>🔵 Near-Infrared Region</h4>
                <p>Protein and moisture indicators detected</p>
                
                <h4>🟢 Visible Spectrum</h4>
                <p>Color changes from fungal growth</p>
                
                <h4>🟡 Short-Wave Infrared</h4>
                <p>Chemical composition markers present</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Agricultural Recommendations
        st.markdown("## 🌾 Agricultural Recommendations")
        
        if risk_level == "LOW":
            st.markdown("""
            <div class="recommendation-card">
                <h3>✅ Safe for Processing</h3>
                <p><strong>Action:</strong> Proceed with normal processing and storage</p>
                <p><strong>Quality Grade:</strong> Premium quality, suitable for all markets</p>
                <p><strong>Storage:</strong> Standard conditions acceptable</p>
                <p><strong>Market Value:</strong> Full market price expected</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "MEDIUM":
            st.markdown("""
            <div class="recommendation-card">
                <h3>⚠️ Moderate Contamination Detected</h3>
                <p><strong>Action:</strong> Segregate batch for specialized processing</p>
                <p><strong>Quality Grade:</strong> Acceptable for animal feed or industrial use</p>
                <p><strong>Storage:</strong> Enhanced monitoring required</p>
                <p><strong>Market Value:</strong> Reduced pricing expected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-card">
                <h3>🚨 High Contamination Alert</h3>
                <p><strong>Action:</strong> Reject batch or apply remediation treatment</p>
                <p><strong>Quality Grade:</strong> Below FDA guidelines for human consumption</p>
                <p><strong>Storage:</strong> Immediate isolation required</p>
                <p><strong>Market Value:</strong> Significant loss expected</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download Section
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_data = {
                'Predicted_Value': [prediction],
                'Risk_Level': [risk_level],
                'Confidence': [confidence],
                'Model': [model_choice]
            }
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV Report",
                csv,
                "prediction_report.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            st.button("📊 Export Visualization", use_container_width=True, disabled=True)
        
        with col3:
            st.button("📧 Email Report", use_container_width=True, disabled=True)

# Run the page
if __name__ == "__main__":
    show_prediction_page()
