import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from Stepwise_selection import stepwise_selection

#### Load data
df = pd.read_csv('C:/Users/Vera.Hollink/OneDrive - Hogeschool Inholland/Documents/My Documents/Machine Learning/Data sets/iris.csv', header = 0)

### STATSMODELS Build univariate regression model to predict petal_width based on sepal_width
# statsmodels.OLS is Ordinary Least Squares
X = df['sepal_width']
X = sm.add_constant(X)
Y = df['petal_width']
model = sm.OLS(Y,X)
results = model.fit()

# print parameters and tvalues
print("########### Coefficients Statsmodels model ###########")
print(results.params)
input("\nPress Enter to continue...")

# print full results
print("\n\n########### Model summary ###########")
print(results.summary())
input("\nPress Enter to continue...")

# Use the model to predict the petal_width of an iris with sepal_width 2.5
print("\n\n########### Prediction ###########")
prediction = results.predict([1,2.5])
print("Predicted petal_width:\n",prediction)

# compute the confidence interval.
# mean_ci_lower/upper is the confidence interval of the mean value
# obs_ci_lower/upper is the confidence interval of the predicted value
prediction_conf = results.get_prediction([1,2.5]).summary_frame(alpha=0.05)
print("\nPredicted petal_width with prediction interval:\n",prediction_conf)
input("\nPress Enter to continue...")

### SCIKIT-LEARN Build univariate regression model to predict petal_width based on sepal_width
# statsmodels.OLS is Ordinary Least Squares
model = LinearRegression().fit(df[['sepal_width']], Y)
print("########### Coefficients Scikit-learn model ###########")
print("constant\t",model.intercept_)
print("sepal_width\t",model.coef_[0])
input("\nPress Enter to continue...")

# Use the model to predict the petal_width of an iris with sepal_width 2.5
print("\n\n########### Prediction ###########")
prediction = model.predict([[2.5]])
print("Predicted petal_width:\n",prediction)

### Build multivariate regression model to predict petal_with based on sepal_width, sepal_length and petal_length
# statsmodels.OLS is Ordinary Least Squares
X = df[['sepal_width','sepal_length','petal_length']]
X = sm.add_constant(X)
Y = df['petal_width']
model = sm.OLS(Y,X)
results = model.fit()

# print parameters and tvalues
print("\n\n########### Coefficients: ###########")
print(results.params)
input("\nPress Enter to continue...")

# print full results
print("\n\n########### Model summary ########### ")
print(results.summary())
input("\nPress Enter to continue...")

# Variable selection
print("\n\n########### Variable selection ########### ")
results = stepwise_selection(X,Y,threshold_in=0.01, threshold_out = 0.01)

