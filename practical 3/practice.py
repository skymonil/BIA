 # 1. Import libraries
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 2. Load DS
df_sal = pd.read_csv('Salary_Data.csv',delimiter=',')

# 3. Display Data
print(df_sal.head())
print(df_sal.describe())

# 4. Create Scatter Plot
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'])

# 5. Seperate X and Y
X = df_sal[['YearsExperience']]
Y = df_sal[['Salary']]

# 6. split data into testing and training
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20,random_state=42)

# 7. Create Regresssion Model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 8. Preict Output

Y_pred_train = regressor.predict(X_train)
Y_pred_test = regressor.predict(X_test)

#9. Plot Training Result
plt.scatter(X_train,Y_train)
plt.plot(X_train,Y_pred_train)

print("Slope (Coefficient):", regressor.coef_)
print("Intercept:", regressor.intercept_)

plt.show(block=True)
