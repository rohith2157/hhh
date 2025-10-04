"""
Simple Linear Regression Model: Sales vs Advertising Analysis
=============================================================

Build a Simple Linear Regression model to study the linear relationship 
between Sales and Advertising dataset for a dietary weight control product.

Implementation in Python Using Scikit-Learn:
1. Import required libraries
2. Load and explore the dataset  
3. Split the dataset into training and testing sets
4. Train the model
5. Make predictions
6. Evaluate the model
"""

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SIMPLE LINEAR REGRESSION: SALES vs ADVERTISING ANALYSIS")
print("="*60)
print("Step 1: ✅ Required libraries imported successfully")
print("- NumPy: Numerical computations")
print("- Pandas: Data manipulation")  
print("- Matplotlib & Seaborn: Data visualization")
print("- Scikit-learn: Machine learning algorithms")
print()

# Step 2: Load and explore the dataset
print("Step 2: 📊 Loading and exploring the dataset")
print("-" * 50)

# Generate sample dataset for dietary weight control product
# Sales vs Advertising relationship
np.random.seed(42)  # For reproducibility

# Generate advertising spend data (in thousands of dollars)
n_samples = 100
advertising_spend = np.random.uniform(10, 100, n_samples)

# Generate sales data with linear relationship + some noise
# Assumption: Sales = 2.5 * Advertising + 15 + noise
true_slope = 2.5
true_intercept = 15
noise = np.random.normal(0, 8, n_samples)
sales = true_slope * advertising_spend + true_intercept + noise

# Create DataFrame
data = pd.DataFrame({
    'Advertising_Spend': advertising_spend,
    'Sales': sales
})

print(f"✅ Dataset created successfully!")
print(f"   - Dataset shape: {data.shape}")
print(f"   - Features: {list(data.columns)}")
print()

# Dataset exploration
print("📋 Dataset Information:")
print(data.info())
print()

print("📈 Statistical Summary:")
print(data.describe())
print()

print("🔍 First 5 rows of the dataset:")
print(data.head())
print()

# Check for missing values
print("🔎 Missing values check:")
print(data.isnull().sum())
print()

# Correlation analysis
correlation = data['Advertising_Spend'].corr(data['Sales'])
print(f"📊 Correlation between Advertising Spend and Sales: {correlation:.4f}")
print()

# Step 3: Split the dataset into training and testing sets
print("Step 3: 🔄 Splitting dataset into training and testing sets")
print("-" * 50)

# Define features (X) and target variable (y)
X = data[['Advertising_Spend']]  # Independent variable (features)
y = data['Sales']                # Dependent variable (target)

print("✅ Features (X) and target (y) defined:")
print(f"   - X shape: {X.shape}")
print(f"   - y shape: {y.shape}")
print()

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Dataset split successfully:")
print(f"   - Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"   - Testing set:  X_test {X_test.shape}, y_test {y_test.shape}")
print(f"   - Training ratio: {len(X_train)/len(X)*100:.1f}%")
print(f"   - Testing ratio:  {len(X_test)/len(X)*100:.1f}%")
print()

# Step 4: Train the model
print("Step 4: 🤖 Training the Linear Regression model")
print("-" * 50)

# Create Linear Regression model
model = LinearRegression()

print("✅ Linear Regression model created")
print("   - Algorithm: Ordinary Least Squares (OLS)")
print("   - Type: Simple Linear Regression")
print()

# Fit the model to training data
print("🔄 Training the model...")
model.fit(X_train, y_train)

# Extract model parameters
slope = model.coef_[0]
intercept = model.intercept_

print("✅ Model training completed successfully!")
print()
print("📋 Model Parameters:")
print(f"   - Slope (β₁):     {slope:.4f}")
print(f"   - Intercept (β₀): {intercept:.4f}")
print()
print("📐 Linear Regression Equation:")
print(f"   Sales = {slope:.4f} × Advertising_Spend + {intercept:.4f}")
print()
print("💡 Business Interpretation:")
print("   - For every $1,000 increase in advertising spend,")
print(f"     sales increase by ${slope*1000:.0f}")
print(f"   - Base sales (when advertising = 0): ${intercept*1000:.0f}")
print()

# Step 5: Make predictions
print("Step 5: 🔮 Making predictions")
print("-" * 50)

# Make predictions on training set
y_train_pred = model.predict(X_train)
print("✅ Predictions made on training set")
print(f"   - Number of training predictions: {len(y_train_pred)}")

# Make predictions on testing set
y_test_pred = model.predict(X_test)
print("✅ Predictions made on testing set")
print(f"   - Number of testing predictions: {len(y_test_pred)}")
print()

# Show sample predictions vs actual values
print("📋 Sample Predictions vs Actual (Test Set):")
print("   Advertising  |  Actual Sales  |  Predicted Sales  |  Difference")
print("   " + "-"*65)
for i in range(min(5, len(X_test))):
    adv = X_test.iloc[i, 0]
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    diff = actual - predicted
    print(f"   ${adv:8.2f}    |   ${actual:8.2f}     |    ${predicted:8.2f}      |   {diff:+6.2f}")
print()

# Step 6: Evaluate the model
print("Step 6: 📊 Evaluating the model performance")
print("-" * 50)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for testing set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("📈 TRAINING SET PERFORMANCE:")
print(f"   - Mean Squared Error (MSE):     {train_mse:.4f}")
print(f"   - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"   - Mean Absolute Error (MAE):    {train_mae:.4f}")
print(f"   - R-squared (R²):              {train_r2:.4f}")
print()

print("🎯 TESTING SET PERFORMANCE:")
print(f"   - Mean Squared Error (MSE):     {test_mse:.4f}")
print(f"   - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"   - Mean Absolute Error (MAE):    {test_mae:.4f}")
print(f"   - R-squared (R²):              {test_r2:.4f}")
print()

# Model interpretation
print("🔍 MODEL EVALUATION SUMMARY:")
print(f"   - Model explains {test_r2*100:.1f}% of variance in sales data")
if test_r2 > 0.8:
    performance = "Excellent"
elif test_r2 > 0.6:
    performance = "Good" 
elif test_r2 > 0.4:
    performance = "Moderate"
else:
    performance = "Poor"

print(f"   - Performance rating: {performance}")
print(f"   - Average prediction error: ±${test_rmse:.2f}K")
print()

# Business insights
print("💼 BUSINESS INSIGHTS:")
print("   - Strong linear relationship between advertising and sales")
print(f"   - Investment in advertising shows {slope:.2f}x return in sales")
print("   - Model is suitable for sales forecasting and budget planning")
print("   - Recommend continued investment in advertising campaigns")
print()

print("="*60)
print("✅ SIMPLE LINEAR REGRESSION ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*60)