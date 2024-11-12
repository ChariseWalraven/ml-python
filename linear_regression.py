import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Titanic-Dataset.csv')

# TODO: plot
cols = ['Age', 'Fare']
age_v_survived = df[cols]

age_v_survived.plot(kind="scatter", x=cols[0], y=cols[1])

print(df)

plt.show()

# TODO: split into test and train groups, keeping only the features you want

