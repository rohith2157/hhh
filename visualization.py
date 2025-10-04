"""
Visualization Script for Simple Linear Regression Analysis
==========================================================

This script creates visualizations to complement the linear regression analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Generate the same dataset (using same seed for consistency)
np.random.seed(42)
n_samples = 100
advertising_spend = np.random.uniform(10, 100, n_samples)
true_slope = 2.5
true_intercept = 15
noise = np.random.normal(0, 8, n_samples)
sales = true_slope * advertising_spend + true_intercept + noise

# Create DataFrame
data = pd.DataFrame({
    'Advertising_Spend': advertising_spend,
    'Sales': sales
})

# Split and train model
X = data[['Advertising_Spend']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Simple Linear Regression: Sales vs Advertising Analysis', fontsize=16, fontweight='bold')

# 1. Scatter plot with regression line
axes[0,0].scatter(data['Advertising_Spend'], data['Sales'], alpha=0.6, color='blue', s=50)
x_line = np.linspace(data['Advertising_Spend'].min(), data['Advertising_Spend'].max(), 100)
y_line = model.predict(x_line.reshape(-1, 1))
axes[0,0].plot(x_line, y_line, color='red', linewidth=2, label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
axes[0,0].set_xlabel('Advertising Spend (thousands $)')
axes[0,0].set_ylabel('Sales (thousands $)')
axes[0,0].set_title('Sales vs Advertising Spend with Regression Line')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Residuals plot
residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Predicted Sales (thousands $)')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residuals Plot')
axes[0,1].grid(True, alpha=0.3)

# 3. Actual vs Predicted
axes[1,0].scatter(y_test, y_pred, alpha=0.6, color='purple', s=50)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[1,0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Actual Sales (thousands $)')
axes[1,0].set_ylabel('Predicted Sales (thousands $)')
axes[1,0].set_title(f'Actual vs Predicted Sales (R² = {r2_score(y_test, y_pred):.3f})')
axes[1,0].grid(True, alpha=0.3)

# 4. Distribution of residuals
axes[1,1].hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].set_xlabel('Residuals')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Residuals')
axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)
print(f"📊 Dataset: {len(data)} samples of dietary weight control product")
print(f"📈 Correlation: {data['Advertising_Spend'].corr(data['Sales']):.4f}")
print(f"🎯 Model R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"📉 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"💡 Business Insight: Every $1K in advertising → ${model.coef_[0]*1000:.0f} in sales")
print("="*60)