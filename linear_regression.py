import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/libraries_in_Poland.csv', sep=";")

# Select features to use for input/prediction
x_cols = ['Number_of_libraries', 'Amount_of_books', 'Number_of_loans', 'Number_of_librarians',
          'Number_with_accessibility_features',
          'Number_with_accessible_entrance',
          'Number_with_accessible_interior', ]
y_col = 'Number_of_readers'

# get input (x) and output (y)
libraries_x = df[x_cols]
libraries_y = df[[y_col]]

# split into train and test
x_train = libraries_x[0:-3]
y_train = libraries_y[0:-3]

x_test = libraries_x[-3:]
y_test = libraries_y[-3:]

# regression_model = linear_model.LinearRegression()

# regression_model.fit(x_train, y_train)

# libraries_y_predictions = regression_model.predict(x_test)
#
# # ?
# print("Coefficients: \n", regression_model.coef_)
# # how wrong the results are
# print("Mean squared error: %.2f" % mean_squared_error(y_test, libraries_y_predictions))
# # how accurate the predictions are
# print("Coefficient of determination (r2 score): %.2f" % r2_score(y_test, libraries_y_predictions))

# try to plot all the subplots maybe just start with them vertically stacked? easier, no?
def plot_features():
    cols = 3
    rows = math.ceil(len(x_cols) / cols)
    col = 0
    row = 0
    fig, axes = plt.subplots(rows, cols)
    fig.suptitle('Feature n and number of readers')
    # add space so you can see the titles of the axes properly
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    num_extra_axes = axes.size - len(x_cols)
    if num_extra_axes > 0:
        print(f'{num_extra_axes} extra axes/axis added')
        for ax in axes[-1][-num_extra_axes:]:
            fig.delaxes(ax)
    for i, col_name in enumerate(x_cols):

        print("i:", i, "row:", row, "col:", col, col_name)
        ax: plt.Axes = axes[row, col]
        ax.scatter(df[[col_name]], df[[y_col]])
        ax.set_xlabel(col_name)
        ax.set_ylabel(y_col)

        if col == 0:
            col = 1
        elif col == 1:
            col = 2
        else:
            row += 1
            col = 0
    # show full screen, since we have quite a few plots
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


# plot_features()
