from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def plot_columns(columns):
    """Plots the given 2 columns on a scatter plot."""
    f1_f2 = df[columns]
    f1_f2.plot(kind="scatter", x=columns[0], y=columns[1])
    plt.show()

df = pd.read_csv('data/libraries_in_Poland.csv', sep=";")

# Select features to use for input/prediction
cols = ['Number_of_libraries', 'Number_of_librarians']


# get input (x) and output (y)
libraries_x = df[[cols[0]]]
libraries_y = df[[cols[1]]]

# NOTE: I get an error relating to passing a pandas Series instead of a DataFrame with a single row or column.
#       When I google this, I get the recommendation to use the following syntax: df.iloc[:,0:1], but am not
#       sure why or what the difference is

# split into train and test
x_train = libraries_x[0:-3]
y_train = libraries_y[0:-3]

x_test = libraries_x[-3:]
y_test = libraries_y[-3:]

regression_model = linear_model.LinearRegression()

regression_model.fit(x_train, y_train)

libraries_y_predictions = regression_model.predict(x_test)

print("Coefficients: \n", regression_model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, libraries_y_predictions))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination (r2 score): %.2f" % r2_score(y_test, libraries_y_predictions))

plt.scatter(x_test, y_test, color="black", label="Test data")
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.plot(x_test, libraries_y_predictions, color="blue", linewidth=3, label="predictions (Linear Regression)")

plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.title("# librarians vs libraries per region in Poland")

plt.show()

# plot_columns(cols)
