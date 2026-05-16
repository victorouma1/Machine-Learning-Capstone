import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

data = pd.read_csv('yield_df.csv')
data.drop(columns = 'Unnamed: 0', inplace = True)

numeric_cols = data.select_dtypes(include = ['int64','float64']).columns
for col in numeric_cols:
    plt.figure()
    sns.histplot(data[col], kde = True)
    plt.show()

sns.heatmap(data.corr(numeric_only = True), annot = True, cmap = 'coolwarm')
plt.title('Correlation heatmap')
plt.show()

sns.pairplot(
    data, 
    hue='Item', 
    vars=['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'],
    palette='viridis',
    diag_kind='kde',
    plot_kws={'alpha': 0.6}
)

plt.show()

encoder = TargetEncoder(cols=['Area', 'Item'])
data[['Area','Item']] = encoder.fit_transform(data[['Area', 'Item']], data['hg/ha_yield'])

X = data.drop(columns = 'hg/ha_yield')
y = data['hg/ha_yield']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

train_preds = lr_model.predict(X_train)
test_preds = lr_model.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds)}")

y_pred = lr_model.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()

residuals = y_test - test_preds

plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='red', linestyle='--')
sns.scatterplot(x=test_preds, y=residuals, alpha=0.6)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Errors)')
plt.title('Residual Plot')

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_poly,y_train)

y_poly_pred = poly_model.predict(poly.transform(X_test))

train_preds_poly = poly_model.predict(X_poly)
test_preds_poly = poly_model.predict(poly.transform(X_test))

print(f"Training R2 score: {r2_score(y_train, train_preds_poly)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_poly)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_poly)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_poly)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_poly)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_poly)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_poly)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_poly)}")

residuals_1 = y_test - test_preds_poly

plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='red', linestyle='--')
sns.scatterplot(x=test_preds_poly, y=residuals_1, alpha=0.6)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Errors)')
plt.title('Residual Plot')

train_sizes, train_scores, val_scores = learning_curve(
    poly_model,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")

plt.title("Learning Curve - Polynomial")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid()
plt.show()

ridge_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler_post_poly', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

train_preds_ridge = ridge_model.predict(X_train)
test_preds_ridge = ridge_model.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds_ridge)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_ridge)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_ridge)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_ridge)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_ridge)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_ridge)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_ridge)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_ridge)}")


train_sizes, train_scores, val_scores = learning_curve(
    ridge_model,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")

plt.title("Learning Curve - Polynomial Ridge")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid()
plt.show()

lasso_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler_post_poly', StandardScaler()),
    ('lasso', Lasso(alpha=1.0))
])

lasso_model.fit(X_train, y_train)

train_preds_lasso = lasso_model.predict(X_train)
test_preds_lasso = lasso_model.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds_lasso)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_lasso)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_lasso)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_lasso)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_lasso)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_lasso)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_lasso)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_lasso)}")

dt_regr_model = DecisionTreeRegressor(max_depth=5)
dt_regr_model.fit(X_train, y_train)

train_preds_dt = dt_regr_model.predict(X_train)
test_preds_dt = dt_regr_model.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds_dt)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_dt)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_dt)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_dt)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_dt)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_dt)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_dt)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_dt)}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_preds_rf = rf_model.predict(X_train)
test_preds_rf = rf_model.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds_rf)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_rf)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_rf)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_rf)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_rf)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_rf)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_rf)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_rf)}")

train_sizes, train_scores, val_scores = learning_curve(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")

plt.title("Learning Curve - Random Forest")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid()
plt.show()

pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler_post_poly', StandardScaler()),
    ('ridge', Ridge())
])
param_dist = {
    'poly__degree': randint(1, 10),
    'ridge__alpha': uniform(0.01, 10)
}

random_search = RandomizedSearchCV(
    ridge_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Parameters (Random Search):")
print(random_search.best_params_)

print("Best CV Score:", random_search.best_score_)

ridge_model_tuned = Pipeline([
    ('poly', PolynomialFeatures(degree=7)),
    ('scaler_post_poly', StandardScaler()),
    ('ridge', Ridge(alpha=0.591))
])

ridge_model_tuned.fit(X_train, y_train)

train_preds_ridge_2 = ridge_model_tuned.predict(X_train)
test_preds_ridge_2 = ridge_model_tuned.predict(X_test)

print(f"Training R2 score: {r2_score(y_train, train_preds_ridge_2)}")
print(f"Testing R2 score: {r2_score(y_test, test_preds_ridge_2)}")

print(f"Training MAE score: {mean_absolute_error(y_train, train_preds_ridge_2)}")
print(f"Testing MAE score: {mean_absolute_error(y_test, test_preds_ridge_2)}")

print(f"Training MSE score: {mean_squared_error(y_train, train_preds_ridge_2)}")
print(f"Testing MSE score: {mean_squared_error(y_test, test_preds_ridge_2)}")

print(f"Training RMSE score: {root_mean_squared_error(y_train, train_preds_ridge_2)}")
print(f"Testing RMSE score: {root_mean_squared_error(y_test, test_preds_ridge_2)}")

train_sizes, train_scores, val_scores = learning_curve(
    ridge_model_tuned,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")

plt.title("Learning Curve - Polynomial Ridge Tuned")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid()
plt.show()