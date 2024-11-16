import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/libraries_in_Poland.csv', sep=";")

# Select features to use for input/prediction
cols = ['Number_of_libraries', 'Number_of_librarians']


# get input (x) and output (y)
libraries_x = df[[cols[0]]]
libraries_y = df[[cols[1]]]

# split into train and test
x_train = libraries_x[0:-3]
y_train = libraries_y[0:-3]

x_test = libraries_x[-3:]
y_test = libraries_y[-3:]

regression_model = linear_model.LinearRegression()

regression_model.fit(x_train, y_train)

libraries_y_predictions = regression_model.predict(x_test)

# ?
print("Coefficients: \n", regression_model.coef_)
# how wrong the results are
print("Mean squared error: %.2f" % mean_squared_error(y_test, libraries_y_predictions))
# how accurate the predictions are
print("Coefficient of determination (r2 score): %.2f" % r2_score(y_test, libraries_y_predictions))

plt.scatter(x_test, y_test, color="black", label="Test data")
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.plot(x_test, libraries_y_predictions, color="blue", linewidth=3, label="predictions (Linear Regression)")

plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.title("# librarians vs libraries per region in Poland")

plt.show()
