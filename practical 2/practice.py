import numpy as np 
import pandas as pd

#Load csv
df = pd.read_csv('naive_bayes_excel.csv', delimiter=',')

#Create new cols
df['grade_A'] = np.where(df['G3'] * 5 >= 80, 1, 0)
df['highabsence'] = np.where(df['absences']  >= 10, 1, 0)
df['count'] = 1

# keep only reqd cols
df = df[['highabsence','grade_A','count']]

print("Processed Data:")
print(df)

pt=pd.pivot_table(
      df,
    values='count',
    index='grade_A',
    columns='highabsence',
    aggfunc='count',
    fill_value=0
)

print("\nPivot Table:")
print(pt)

P_A = (pt.loc[1,0] + pt.loc[1,1]) / pt.values.sum()
P_B = (pt.loc[0,1] + pt.loc[1,1]) / pt.values.sum()

P_B_given_A = (pt.loc[1,1]) /  (pt.loc[1,0] + pt.loc[1,1])

P_A_given_B = (P_B_given_A * P_A) / P_B

print("\nProbabilities:")
print("P(A) =", P_A)
print("P(B) =", P_B)
print("P(B|A) =", P_B_given_A)
print("P(A|B) =", P_A_given_B)

