import pandas as pd  # for reading CSV and handling data
import numpy as np  
import matplotlib.pyplot as plt  # for graphs
from sklearn.model_selection import train_test_split # divide data into testing and training
from sklearn.linear_model import LinearRegression

# 1. Load DS
df_sal = pd.read_csv("Salary_Data.csv")

# 2. Display Data
print(df_sal.head()) # shows first 5 rows
print(df_sal.describe()) # gives statistics (mean, min, max etc.)


#  3. Scatter plot (raw data)
#draws Graph of  X-axis → Experience &  Y-axis → Salary
plt.scatter(df_sal['YearsExperience'], df_sal['Salary']) 

#  4. Separate X and Y
# X = input (experience)
# Y = output (Salary)
X = df_sal[['YearsExperience']]
Y = df_sal[['Salary']]

# 5. Split data into training & testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# 6: Create regression model
#👉 Creates linear regression line
# Fits best straight line to training data
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# 7. Predict Salary
# Predicts salary values using regression line
Y_pred_train = regressor.predict(X_train)
Y_pred_test = regressor.predict(X_test)

#8 . Plot Training Result
# Dots = real salaries
# Line = predicted line
plt.scatter(X_train, Y_train)
plt.plot(X_train, Y_pred_train)

print("Slope (Coefficient):", regressor.coef_)
print("Intercept:", regressor.intercept_)

plt.show(block=True)
