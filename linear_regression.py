import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/libraries_in_Poland.csv', sep=";")

# TODO: plot
cols = ['Number_of_libraries', 'Number_of_librarians']
f1_f2 = df[cols]

f1_f2.plot(kind="scatter", x=cols[0], y=cols[1])


# TODO: split into test and train groups, keeping only the features you want
# print(df, f1_f2)
plt.show()

