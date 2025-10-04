# Simple Linear Regression Analysis Report
## Sales vs Advertising for Dietary Weight Control Product

### Project Overview
This project implements a **Simple Linear Regression model** to study the linear relationship between Sales and Advertising dataset for a dietary weight control product using **Scikit-learn** - the popular machine learning library of Python.

### Implementation Steps (Following Your Requirements)

#### ✅ 1. Import Required Libraries
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis  
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms (LinearRegression, train_test_split, metrics)

#### ✅ 2. Load and Explore the Dataset
- **Dataset**: 100 samples of advertising spend vs sales data
- **Features**: Advertising_Spend (independent variable), Sales (dependent variable)
- **Data Quality**: No missing values, clean dataset
- **Correlation**: 0.9940 (very strong positive correlation)

#### ✅ 3. Split Dataset into Training and Testing Sets
- **Training Set**: 80 samples (80%)
- **Testing Set**: 20 samples (20%)
- **Method**: train_test_split with random_state=42 for reproducibility

#### ✅ 4. Train the Model
- **Algorithm**: Linear Regression (Ordinary Least Squares)
- **Model Type**: Simple Linear Regression (one feature)
- **Training**: Fitted on training data using scikit-learn's LinearRegression()

#### ✅ 5. Make Predictions
- **Training Predictions**: 80 predictions on training set
- **Testing Predictions**: 20 predictions on testing set
- **Sample Results**: Model shows excellent prediction accuracy

#### ✅ 6. Evaluate the Model
- **R² Score**: 0.9913 (99.1% variance explained) - Excellent performance
- **RMSE**: 6.47 (average prediction error)
- **MAE**: 4.73 (mean absolute error)
- **Performance Rating**: Excellent

### Key Results

#### Model Equation
```
Sales = 2.4643 × Advertising_Spend + 16.5001
```

#### Business Insights
- **ROI**: For every $1,000 increase in advertising spend, sales increase by $2,464
- **Base Sales**: $16,500 when advertising = 0
- **Strong Relationship**: 99.4% correlation between advertising and sales
- **Model Reliability**: 99.1% of variance explained (R² = 0.9913)

#### Performance Metrics
| Metric | Training Set | Testing Set |
|--------|--------------|-------------|
| R² Score | 0.9872 | 0.9913 |
| RMSE | 7.37 | 6.47 |
| MAE | 5.88 | 4.73 |
| MSE | 54.25 | 41.84 |

### Files Created

1. **`simple_linear_regression.py`** - Main implementation following your 6-step approach
2. **`visualization.py`** - Creates comprehensive plots and visualizations
3. **`linear_regression_report.md`** - This comprehensive report

### Recommendations

1. **Continue Investment**: Strong ROI (2.46x return) supports continued advertising investment
2. **Budget Planning**: Model can be used for sales forecasting and budget allocation  
3. **Scaling**: Consider expanding to multiple features (Multiple Linear Regression)
4. **Monitoring**: Regular model updates with new data to maintain accuracy

### Technical Specifications

- **Python Version**: 3.13+
- **Key Libraries**: scikit-learn, pandas, numpy, matplotlib
- **Algorithm**: Ordinary Least Squares Linear Regression
- **Validation**: Train/test split methodology
- **Reproducibility**: Fixed random seeds for consistent results

---

**Status**: ✅ **COMPLETED SUCCESSFULLY**  
**Performance**: 🌟 **EXCELLENT** (R² = 0.9913)  
**Business Value**: 💰 **HIGH** (Clear ROI insights)