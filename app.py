"""
🔬 SKTime ML Forecasting Dashboard
Professional ML Forecasting with Scikit-learn Compatible Interface

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# SKTime imports
try:
    from sktime.forecasting.arima import ARIMA, AutoARIMA
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.forecasting.compose import make_reduction
    from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV
    from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error
    from sktime.transformations.series.detrend import Detrending
    from sktime.utils.plotting import plot_series
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import sktime
except ImportError as e:
    st.error(f"Error importing SKTime: {e}")
    st.stop()

# ML and feature engineering
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="SKTime ML Forecasting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .ml-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .pipeline-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'accent': '#f093fb',
    'success': '#96CEB4',
    'warning': '#FECA57',
    'error': '#FF9FF3'
}

@st.cache_data
def generate_ml_datasets() -> Dict[str, pd.DataFrame]:
    """Generate datasets suitable for ML forecasting"""
    
    datasets = {}
    
    # 1. Industrial sensor data
    dates = pd.date_range('2022-01-01', '2024-12-01', freq='H')
    np.random.seed(42)
    
    # Base temperature with daily cycles and trends
    hours = np.arange(len(dates))
    base_temp = 20 + 5 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
    seasonal_temp = 10 * np.sin(2 * np.pi * hours / (24 * 365))  # Yearly cycle
    trend = 0.001 * hours  # Slight warming trend
    noise = np.random.normal(0, 1, len(dates))
    
    temperature = base_temp + seasonal_temp + trend + noise
    
    # Sample for performance
    sample_idx = np.arange(0, len(temperature), 24)  # Daily samples
    
    datasets['Industrial_Temperature'] = pd.DataFrame({
        'ds': dates[sample_idx],
        'y': temperature[sample_idx]
    }).set_index('ds')
    
    # 2. Financial returns
    dates_business = pd.date_range('2020-01-01', '2024-12-01', freq='B')
    np.random.seed(123)
    
    # Random walk with volatility clustering
    returns = np.random.normal(0, 0.02, len(dates_business))
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > 0.03:
            returns[i] *= 1.5
    
    # Convert to price levels
    price_levels = 100 * np.exp(np.cumsum(returns))
    
    datasets['Financial_Returns'] = pd.DataFrame({
        'ds': dates_business,
        'y': price_levels
    }).set_index('ds')
    
    # 3. Network traffic
    dates_hourly = pd.date_range('2023-01-01', '2024-06-01', freq='H')
    np.random.seed(456)
    
    # Network usage patterns
    base_traffic = 1000
    hour_pattern = 500 * np.sin(2 * np.pi * dates_hourly.hour / 24)
    day_pattern = 200 * (dates_hourly.weekday < 5).astype(int)  # Weekday effect
    noise_network = np.random.normal(0, 100, len(dates_hourly))
    
    traffic = base_traffic + hour_pattern + day_pattern + noise_network
    traffic = np.maximum(traffic, 100)
    
    # Sample every 6 hours
    sample_idx = np.arange(0, len(traffic), 6)
    
    datasets['Network_Traffic'] = pd.DataFrame({
        'ds': dates_hourly[sample_idx],
        'y': traffic[sample_idx]
    }).set_index('ds')
    
    return datasets

def create_sktime_models() -> Dict[str, object]:
    """Create SKTime models with consistent interface"""
    
    models = {
        'ARIMA': ARIMA(order=(1,1,1)),
        'AutoARIMA': AutoARIMA(sp=12, max_p=3, max_q=3),
        'ExponentialSmoothing': ExponentialSmoothing(trend='add', seasonal='add', sp=12),
        'ThetaForecaster': ThetaForecaster(sp=12),
        'LinearRegression': make_reduction(LinearRegression(), window_length=12),
        'RandomForest': make_reduction(RandomForestRegressor(n_estimators=100), window_length=12),
        'PolynomialTrend': PolynomialTrendForecaster(degree=2)
    }
    
    return models

def perform_model_comparison(data: pd.DataFrame, models: Dict, horizon: int) -> pd.DataFrame:
    """Compare models using SKTime's interface"""
    
    results = []
    
    # Split data
    y_train, y_test = temporal_train_test_split(data['y'], test_size=horizon)
    
    for name, model in models.items():
        try:
            # Fit model
            model.fit(y_train)
            
            # Make prediction
            y_pred = model.predict(fh=np.arange(1, len(y_test) + 1))
            
            # Calculate metrics
            mae_score = mean_absolute_error(y_test, y_pred)
            mse_score = mean_squared_error(y_test, y_pred)
            mape_score = np.mean(np.abs((y_test.values - y_pred.values) / y_test.values)) * 100
            
            results.append({
                'Model': name,
                'MAE': mae_score,
                'MSE': mse_score,
                'MAPE': mape_score
            })
            
        except Exception as e:
            st.warning(f"Error with {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 SKTime ML Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-info">
    🔬 <strong>Professional ML Forecasting with SKTime Framework</strong><br>
    Scikit-learn compatible time series analysis with unified fit(), predict(), transform() API
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ ML Configuration")
        
        # Dataset selection
        datasets = generate_ml_datasets()
        dataset_name = st.selectbox("📊 Select ML Dataset:", list(datasets.keys()))
        data = datasets[dataset_name]
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        available_models = create_sktime_models()
        
        selected_models = {}
        for model_name in available_models.keys():
            if st.checkbox(f"{model_name}", key=f"model_{model_name}"):
                selected_models[model_name] = available_models[model_name]
        
        # Forecasting parameters
        st.markdown("### 📈 Forecasting Setup")
        forecast_horizon = st.slider("🔮 Forecast Horizon:", 6, 48, 12)
        
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        enable_preprocessing = st.checkbox("🔧 Data Preprocessing", value=True)
        enable_cross_validation = st.checkbox("✅ Cross Validation", value=True)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Data visualization
        st.subheader("📊 ML Dataset Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['y'],
            mode='lines',
            name='Time Series',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f"Dataset: {dataset_name}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series properties
        st.markdown("### 📈 Time Series Properties")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("📊 Length", len(data))
        with col_b:
            st.metric("📅 Frequency", str(data.index.freq) if data.index.freq else "Irregular")
        with col_c:
            st.metric("📈 Mean", f"{data['y'].mean():.2f}")
        with col_d:
            st.metric("📊 Std Dev", f"{data['y'].std():.2f}")
    
    with col2:
        # ML pipeline info
        st.subheader("🔬 ML Pipeline")
        
        # Selected models
        if selected_models:
            st.success(f"📊 {len(selected_models)} models selected")
            for model_name in selected_models.keys():
                st.markdown(f"• {model_name}")
        else:
            st.warning("⚠️ Please select at least one model")
        
        # SKTime info
        st.info(f"🔬 SKTime version: {sktime.__version__}")
        
        # Data info
        stationarity = "Likely stationary" if data['y'].diff().std() < data['y'].std() else "Non-stationary"
        st.info(f"📈 {stationarity}")
    
    # Model comparison and forecasting
    if selected_models and st.button("🚀 Run ML Forecasting", type="primary"):
        st.markdown("---")
        st.subheader("🔮 ML Forecasting Results")
        
        with st.spinner("Training ML models and generating forecasts..."):
            # Perform model comparison
            comparison_results = perform_model_comparison(
                data, selected_models, forecast_horizon
            )
            
            if not comparison_results.empty:
                # Display comparison
                st.markdown("### 📊 Model Performance Comparison")
                
                # Sort by MAE
                comparison_results = comparison_results.sort_values('MAE')
                st.dataframe(comparison_results, hide_index=True)
                
                # Best model
                best_model_name = comparison_results.iloc[0]['Model']
                best_mae = comparison_results.iloc[0]['MAE']
                st.success(f"🏆 Best Model: {best_model_name} (MAE: {best_mae:.3f})")
                
                # Generate forecasts visualization
                st.markdown("### 🔮 Forecast Visualization")
                
                # Split data for visualization
                y_train, y_test = temporal_train_test_split(data['y'], test_size=forecast_horizon)
                
                fig = go.Figure()
                
                # Training data
                fig.add_trace(go.Scatter(
                    x=y_train.index,
                    y=y_train.values,
                    mode='lines',
                    name='Training Data',
                    line=dict(color=COLORS['primary'], width=2)
                ))
                
                # Test data
                fig.add_trace(go.Scatter(
                    x=y_test.index,
                    y=y_test.values,
                    mode='lines',
                    name='Actual',
                    line=dict(color=COLORS['secondary'], width=2)
                ))
                
                # Generate forecasts for top 3 models
                colors = [COLORS['accent'], COLORS['success'], COLORS['warning']]
                
                for i, (_, row) in enumerate(comparison_results.head(3).iterrows()):
                    model_name = row['Model']
                    model = selected_models[model_name]
                    
                    try:
                        # Fit and predict
                        model.fit(y_train)
                        y_pred = model.predict(fh=np.arange(1, len(y_test) + 1))
                        
                        fig.add_trace(go.Scatter(
                            x=y_test.index,
                            y=y_pred.values,
                            mode='lines+markers',
                            name=f"{model_name} Forecast",
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                        ))
                        
                    except Exception as e:
                        st.warning(f"Error generating forecast for {model_name}: {str(e)}")
                        continue
                
                fig.update_layout(
                    title="📈 ML Forecasting Comparison",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = comparison_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"sktime_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🔬 <strong>SKTime ML Forecasting</strong> | 
    Built with SKTime framework | 
    <a href="https://github.com/PabloPoletti" target="_blank">GitHub</a> | 
    <a href="mailto:lic.poletti@gmail.com">Contact</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()