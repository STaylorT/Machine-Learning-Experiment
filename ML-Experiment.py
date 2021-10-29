from sklearn.datasets import make_moons, make_circles, make_classification

import numpy as np
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
rng = np.random.RandomState(0)
x += rng.uniform(size=x.shape)
linearly_separable = (x, y)

datasets = [linearly_separable,
            make_moons(n_samples=150, noise=.2, random_state=0),
            make_circles(n_samples=150, noise=.1, factor=.5,
                         random_state=1)
            ]

da1, da2, da3 = datasets[0], datasets[1], datasets[2]

# Dataset1
x1, y1 = da1

plt.scatter(x1[:, 0], x1[:, 1], c=y1)
plt.show()
plt.savefig('da1.pdf')

# Dataset2
x2, y2 = da2

plt.scatter(x2[:, 0], x2[:, 1], c=y2)
plt.show()

plt.savefig('da2.pdf')

# Dataset3
x3, y3 = da3

plt.scatter(x3[:, 0], x3[:, 1], c=y3)
plt.show()
plt.savefig('da3.pdf')

# np.savetxt('da1.csv', x1, delimiter=',')
# np.savetxt('da1_y.csv', y1, fmt='%i', delimiter=',')
#
# np.savetxt('da2.csv', x2, delimiter=',')
# np.savetxt('da2_y.csv', y2, fmt='%i', delimiter=',')
#
# np.savetxt('da3.csv', x3, delimiter=',')
# np.savetxt('da3_y.csv', y3, fmt='%i', delimiter=',')


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset 1
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,
                                                        test_size=.3,
                                                        random_state=42)
clf_ds1 = LogisticRegression(solver='lbfgs',
                             random_state=0).fit(x1_train, y1_train)
pred_y1 = clf_ds1.predict(x1_test)

print("Logistic Reg, D1: ", classification_report(y1_test, pred_y1))

clf_ds1 = LogisticRegression(solver='sag',
                             random_state=0).fit(x1_train, y1_train)
pred_y1 = clf_ds1.predict(x1_test)

print("Logistic Reg ['sag'], D1: ", classification_report(y1_test, pred_y1))


# Dataset 2
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2,
                                                        test_size=.3,
                                                        random_state=42)
clf_ds2 = LogisticRegression(solver='lbfgs',
                             random_state=0).fit(x2_train, y2_train)
pred_y2 = clf_ds2.predict(x2_test)
# test set
print("Logistic Reg, D2: ", classification_report(y2_test, pred_y2))

# training set
train_predict_y2 = clf_ds2.predict(x2_train)
print(classification_report(y2_train, train_predict_y2))


#
# # Dataset 3
# x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3,
#                                                         test_size=.3,
#                                                         random_state=42)
# clf_ds3 = LogisticRegression(solver='lbfgs',
#                              random_state=0).fit(x3_train, y3_train)
# pred_y3 = clf_ds3.predict(x3_test)
# # test set
# print("Logistic Reg, D3: ", classification_report(y3_test, pred_y3))
#
# # training set
# train_predict_y3 = clf_ds3.predict(x3_train)
# print(classification_report(y3_train, train_predict_y3))
#
# # Kernalized SVMs
# #
# #
# #
# from sklearn.svm import SVC
#
# # SVM: RBF Kernel
# clf_kernel = SVC(kernel='RBF', gamma='scale',
#                  random_state=0).fit(x3_train, y3_train)
# pred_y3 = clf_kernel.predict(x3_test)
#
# print("RBF", classification_report(y3_test, pred_y3))
#
# # SVM: Sigmoid Kernel
# clf_kernel = SVC(kernel='sigmoid', gamma='scale',
#                  random_state=0).fit(x3_train, y3_train)
# pred_y3 = clf_kernel.predict(x3_test)
#
# print("Sigmoid", classification_report(y3_test, pred_y3))
#
# # SVM: Polynomial Kernel
# clf_kernel = SVC(kernel='poly', degree=10, gamma='scale',
#                  random_state=0).fit(x3_train, y3_train)
# pred_y3 = clf_kernel.predict(x3_test)
#
# print("Polynomial(10)", classification_report(y3_test, pred_y3))
