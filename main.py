import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

df = pd.read_csv('student-mat.csv',sep=';')
print(df.head())
df = df[['G1', 'G2', 'G3','failures', 'absences']]


x = np.array(df.drop(['G3'], 1))
y = np.array(df['G3'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# for finding best accuracy of model, we can use below code:

# best = 0
#
# for _ in range(10):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         with open('studentmodel.pickle', 'wb') as f:
#             pickle.dump(linear, f)
#
# best = print('best=', best)


newmodel = pickle.load(open('studentmodel.pickle', 'rb'))

print('coefficient:',newmodel.coef_)
print('Intercept:',newmodel.intercept_)

resualt = newmodel.predict(x_test)

for x in range(len(resualt)):
    print(resualt[x], x_test[x], y_test[x])

