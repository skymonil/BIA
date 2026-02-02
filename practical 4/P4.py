import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # matplotlib → graphs   
from sklearn.datasets import load_diabetes # load_diabetes → built-in dataset
from sklearn.model_selection import train_test_split # train_test_split → split data
from sklearn.preprocessing import StandardScaler # StandardScaler → normalize data
from sklearn.linear_model import LogisticRegression # LogisticRegression → classification model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc # metrics → evaluate performance

# 2. Load DS 
diabetes = load_diabetes()

# 3: Separate features and target
# X → input values (health parameters)
# Y → output values (disease measure)
X = diabetes.data
Y = diabetes.target

#4. 4: Convert target into binary class
Y_binary = (Y > np.median(Y)).astype(int)
# Since logistic regression needs 0 or 1, we convert

#Above median → 1 (high diabetes risk)
#Below median → 0 (low diabetes risk)

#5: Split into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_binary, test_size=0.2, random_state=42
)

# 6: Standardize data
# Makes values in same range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#7: Create Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# 8: Predict output
Y_pred = model.predict(X_test)
# Predicts whether person has high or low diabetes risk

# 9. Calculate Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
# Shows percentage of correct predictions

# 10 .Confusion Matrix
confusion_matrix(Y_test, Y_pred)
# show • Correct predictions
# shots Wrong predictions

# 11 ROC Curve
#Shows model performance graphically
Y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_prob)
roc_auc = auc(fpr, tpr)
# ROC curve plots:
# False Positive Rate
# True Positive Rate

# 12: Plot ROC curve
plt.plot(fpr, tpr, label="Model ROC Curve")
plt.plot([0,1], [0,1], '--', label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()