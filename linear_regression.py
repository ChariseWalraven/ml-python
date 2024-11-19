import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/libraries_in_Poland.csv', sep=";")

scaler = MinMaxScaler()



# select features to use for input/prediction
x_cols = ['Number_of_libraries', 'Amount_of_books', 'Number_of_loans', 'Number_of_readers',
          'Number_with_accessibility_features',
          'Number_with_accessible_entrance',
          'Number_with_accessible_interior', ]
y_col = 'Number_of_librarians'

# get input (X) and output (y)
X = df[x_cols]
y = df[[y_col]]

# TODO: use scaling on inputs (use scaled input in model)
print(scaler.fit(X), scaler.data_max_, scaler.data_min_, scaler.data_range_)

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

regression_model = linear_model.LinearRegression()

regression_model.fit(X_train, y_train)

libraries_y_predictions = regression_model.predict(X_test)

# coefficients are the m's in the mx + b formula. They tell us what features are more important,
# given the absolute size of the coefficients.
print("Coefficients: \n", regression_model.coef_)
# how wrong the results are
print("Mean squared error: %.2f" % mean_squared_error(y_test, libraries_y_predictions))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, libraries_y_predictions))
# how accurate the predictions are
print("Coefficient of determination (r2 score): %.2f" % r2_score(y_test, libraries_y_predictions))


def plot_features():
    cols = 3
    rows = math.ceil(len(x_cols) / cols)
    col = 0
    row = 0
    fig, axes = plt.subplots(rows, cols)
    fig.suptitle(f'Feature n and {' '.join(y_col.split('_')).lower()}')
    # add space so you can see the titles of the axes properly
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    num_extra_axes = axes.size - len(x_cols)
    if num_extra_axes > 0:
        for ax in axes[-1][-num_extra_axes:]:
            fig.delaxes(ax)
    for i, col_name in enumerate(x_cols):

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

# ...Wat -> https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlTN62y0Ov_gXsyBmxbWT1RE2E4Bul7ZPZuA&s
