import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import data_prep

df = data_prep.read_and_prepare_data()

print(df[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']].head(20))

df_binary = df[['Calories', 'Fat']]


X = np.array(df_binary['Fat']).reshape(-1,1)
y = np.array(df_binary['Calories']).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')

plt.show()