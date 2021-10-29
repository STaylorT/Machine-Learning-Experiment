print(__doc__)

# Code source: Gael Varoquaux

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

h = .02 # step size
names = ["lbgfgs",
        "sag",
        "saga",
        "newton-cg",
        "liblinear",
        ]

classifiers = [
    LogisticRegression(solver='lbfgs'),
    LogisticRegression(solver='sag'),
    LogisticRegression(solver='saga'),
    LogisticRegression(solver='newton-cg'),
    LogisticRegression(solver='liblinear')
]
# testing across regularization parameter
classifiers1 = [
    LogisticRegression(solver='lbfgs', C=.0001),
    LogisticRegression(solver='lbfgs', C=.001),
    LogisticRegression(solver='lbfgs', C=.01),
    LogisticRegression(solver='lbfgs', C=.1),
    LogisticRegression(solver='lbfgs', C= 1),

]

x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(0)
x += rng.uniform(size=x.shape)
linearly_separable = (x,y)

datasets = [linearly_separable#,
            # make_moons(noise=.2, random_state=0)  #,
#             make_circles(noise=.3, factor=.5, random_state=1)
            ]

figure = plt.figure(figsize=(27,9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test parts
    x, y = ds
    x = StandardScaler().fit_transform(x) # Fit to data, then transform
    x_train, x_test, y_train, y_test = \
        train_test_split(x,y, test_size=.4, random_state=42)
    x_min, x_max = x[:,0].min() - .5, x[:,0].max() + .5
    y_min, y_max  = x[:,1].min() - .5, x[:,1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap('#f00', '#00f')
    ax = plt.subplot(len(datasets), len(classifiers1) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training pts
    ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_bright, alpha=.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers1):
        ax = plt.subplot(len(datasets), len(classifiers1) + 1, i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)

        #Plot the decision boundary. assign a color to each point in mesh
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

        # Put result into color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        size = 15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.show()