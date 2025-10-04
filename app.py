"""
Streamlit Web Application for Linear Regression Analysis
========================================================

A web-based interface for the Sales vs Advertising Linear Regression model.
This can be deployed to Streamlit Cloud, Heroku, or other platforms.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Linear Regression Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_samples=100, noise_level=8, seed=42):
    """Generate sample Sales vs Advertising data."""
    np.random.seed(seed)
    
    # Generate advertising spend data (in thousands)
    advertising = np.random.uniform(10, 100, n_samples)
    
    # Generate sales data with linear relationship + noise
    sales = 2.5 * advertising + 15 + np.random.normal(0, noise_level, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Advertising': advertising,
        'Sales': sales
    })
    
    return data

def train_model(data, test_size=0.2, random_state=42):
    """Train the linear regression model."""
    X = data[['Advertising']]
    y = data['Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Linear Regression: Sales vs Advertising</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive web application demonstrates Simple Linear Regression analysis 
    for studying the relationship between advertising spend and sales for a dietary weight control product.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Model Parameters")
    
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100, 10)
    noise_level = st.sidebar.slider("Noise Level", 1, 20, 8, 1)
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5) / 100
    random_seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
    
    # Generate data
    data = generate_sample_data(n_samples, noise_level, random_seed)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(
        data, test_size, random_seed
    )
    
    # Calculate metrics
    train_mse, train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_pred_train)
    test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(y_test, y_pred_test)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Visualization")
        
        # Create scatter plot with regression line
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=data['Advertising'], 
            y=data['Sales'],
            mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add regression line
        x_line = np.linspace(data['Advertising'].min(), data['Advertising'].max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=x_line, 
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Sales vs Advertising with Regression Line",
            xaxis_title="Advertising Spend (thousands)",
            yaxis_title="Sales (thousands)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.subheader("📊 Data Summary")
        st.dataframe(data.describe())
        
    with col2:
        st.subheader("🎯 Model Performance")
        
        # Performance metrics
        col_train, col_test = st.columns(2)
        
        with col_train:
            st.markdown("**Training Set**")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("RMSE", f"{train_rmse:.2f}")
            st.metric("MAE", f"{train_mae:.2f}")
            
        with col_test:
            st.markdown("**Test Set**")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("RMSE", f"{test_rmse:.2f}")
            st.metric("MAE", f"{test_mae:.2f}")
        
        # Model coefficients
        st.subheader("🔍 Model Details")
        st.write(f"**Slope (β₁):** {model.coef_[0]:.4f}")
        st.write(f"**Intercept (β₀):** {model.intercept_:.4f}")
        st.write(f"**Equation:** Sales = {model.coef_[0]:.4f} × Advertising + {model.intercept_:.4f}")
        
        # Business insights
        st.subheader("💡 Business Insights")
        if model.coef_[0] > 0:
            st.success(f"✅ For every $1,000 increase in advertising spend, sales increase by ${model.coef_[0]*1000:.0f}")
        
        if test_r2 > 0.7:
            st.success(f"✅ Strong relationship (R² = {test_r2:.3f})")
        elif test_r2 > 0.5:
            st.warning(f"⚠️ Moderate relationship (R² = {test_r2:.3f})")
        else:
            st.error(f"❌ Weak relationship (R² = {test_r2:.3f})")
    
    # Additional visualizations
    st.subheader("📈 Additional Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Residuals plot
        residuals = y_test - y_pred_test
        fig_residuals = px.scatter(
            x=y_pred_test, 
            y=residuals,
            title="Residuals vs Predicted Values",
            labels={'x': 'Predicted Sales', 'y': 'Residuals'}
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col4:
        # Distribution comparison
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=y_test, name="Actual Sales", opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=y_pred_test, name="Predicted Sales", opacity=0.7))
        fig_dist.update_layout(
            title="Distribution: Actual vs Predicted Sales",
            xaxis_title="Sales (thousands)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Data table
    if st.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Data")
        st.dataframe(data)
    
    # PREDICTION SECTION - This is what you were looking for!
    st.markdown("---")
    st.subheader("🔮 **TRY IT YOURSELF - Make Your Own Predictions!**")
    st.markdown("### Enter your advertising budget and get sales prediction:")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### 💰 **INPUT (What YOU Enter)**")
        advertising_budget = st.number_input(
            "Enter Advertising Budget ($1000s):",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0,
            help="Enter how much you want to spend on advertising (in thousands of dollars)"
        )
        
        if st.button("🚀 **PREDICT SALES**", type="primary"):
            # Make prediction using the trained model
            predicted_sales = model.predict(np.array([[advertising_budget]]))[0]
            profit = predicted_sales - advertising_budget
            roi = (profit / advertising_budget) * 100 if advertising_budget > 0 else 0
            
            with col_output:
                st.markdown("#### 📈 **OUTPUT (Prediction Results)**")
                st.success(f"💵 **Predicted Sales: ${predicted_sales:.2f}K**")
                st.info(f"💰 **Profit: ${profit:.2f}K**")
                st.info(f"📊 **ROI: {roi:.1f}%**")
                
                # Business advice
                if roi > 100:
                    st.success("✅ **EXCELLENT INVESTMENT!** High returns expected.")
                elif roi > 50:
                    st.warning("⚠️ **GOOD INVESTMENT** - Decent returns.")
                else:
                    st.error("❌ **RISKY INVESTMENT** - Low returns.")
    
    # Example scenarios
    st.markdown("---")
    st.subheader("💡 **Quick Examples - Try These!**")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.markdown("**💵 Small Budget**")
        st.markdown("Input: $20K advertising")
        pred_20 = model.predict(np.array([[20]]))[0]
        st.markdown(f"Output: ${pred_20:.1f}K sales")
        st.markdown(f"Profit: ${pred_20-20:.1f}K")
    
    with example_col2:
        st.markdown("**💰 Medium Budget**")
        st.markdown("Input: $50K advertising")  
        pred_50 = model.predict(np.array([[50]]))[0]
        st.markdown(f"Output: ${pred_50:.1f}K sales")
        st.markdown(f"Profit: ${pred_50-50:.1f}K")
    
    with example_col3:
        st.markdown("**🚀 Large Budget**")
        st.markdown("Input: $100K advertising")
        pred_100 = model.predict(np.array([[100]]))[0]
        st.markdown(f"Output: ${pred_100:.1f}K sales")
        st.markdown(f"Profit: ${pred_100-100:.1f}K")

if __name__ == "__main__":
    main()
