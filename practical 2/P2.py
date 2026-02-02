import pandas as pd
import numpy as np

#Load CSV file
df = pd.read_csv('./naive_bayes_excel.csv', delimiter=',')

# Create new columns
df['grade_A'] = np.where(df['G3'] * 5 >= 80, 1, 0) # Takes student marks (G3) and converts to binary A grade by multiplying by 5 and checking if >= 80 put 1 else 0
df['highabsences'] = np.where(df['absences'] >= 10, 1, 0) # Creates binary column for high absences 
df['count'] = 1 # Adds a column with value 1 for every row


# Keep required columns only
df = df[['grade_A', 'highabsences', 'count']]

print("Processed Data:")
print(df)

# Pivot Table
pt = pd.pivot_table(
    df,
    values='count',
    index='grade_A',
    columns='highabsences',
    aggfunc='count',
    fill_value=0
)

print("\nPivot Table:")
print(pt)

# Prior Probability P(A) where A = grade_A = 1
P_A = (pt.loc[1,0] + pt.loc[1,1]) / pt.values.sum()

# Prior Probability P(B) where B = highabsences = 1
P_B = (pt.loc[0, 1] + pt.loc[1, 1]) / pt.values.sum()

# Likelihood P(B|A)
P_B_given_A = pt.loc[1, 1] / (pt.loc[1, 0] + pt.loc[1, 1])

# Posterior Probability using Bayes Theorem
P_A_given_B = (P_B_given_A * P_A) / P_B

print("\nProbabilities:")
print("P(A) =", P_A)
print("P(B) =", P_B)
print("P(B|A) =", P_B_given_A)
print("P(A|B) =", P_A_given_B)
