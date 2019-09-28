import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
sns.set_palette("RdBu")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings('ignore')

###############################################
#
#       Reference:
#           Linear Classification <https://qiita.com/fujin/items/f5656afc8a40fcf55386>
#           
###############################################

TOP_DIR = '..'
col = 400
row = 100
bias = 0.289

fig = plt.figure(figsize=(10,10))

def run_extra_tree_classifier(X, Y ,mesh_data):
    print("""
--------------------------------

        Extremely Randomized Trees

--------------------------------""")
    clf = ExtraTreesClassifier(random_state=0)
    clf = clf.fit(X, Y)
    
    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(clf.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[3] , linewidth=0, s=2)
    fig.axes[3].set_title('Extremely Randomized Trees')

def run_random_forest(X, Y, mesh_data):
    print("""
--------------------------------

        Random Forest

--------------------------------""")
    clf = RandomForestClassifier(random_state=0)
    clf = clf.fit(X, Y)
    
    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(clf.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[2] , linewidth=0, s=2)
    fig.axes[2].set_title('Random Forest')

def run_decision_tree(X, Y, mesh_data):
    print("""
--------------------------------

        Decision Tree

--------------------------------""")
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(X, Y)
    
    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(clf.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[1] , linewidth=0, s=2)
    fig.axes[1].set_title('Decision Tree')


if __name__ == "__main__":
    dim = 10000
    noise =  np.random.rand(dim) * 0.5 - 0.25

    X1 = np.linspace(0, np.pi, num=dim)
    Y1 = np.sin(X1) + noise - bias

    X2 = np.linspace(0.5 * np.pi, 1.5 * np.pi, num=dim)
    Y2 = -1 * np.sin(X2 - 0.5 * np.pi) + noise  + bias
    data = np.vstack([X1, Y1, X2, Y2])

    df = pd.DataFrame(data.T, columns=['x1', 'y1', 'x2', 'y2'])

    for i in [1,2,3,4]:
        temp = fig.add_subplot(2,2,i)
        sns.scatterplot(x="x1", y="y1", data=df, ax=temp, linewidth=0, s=2)
        sns.scatterplot(x="x2", y="y2", data=df, ax=temp, linewidth=0, s=2)
        temp.set_xlim([0, 2 * np.pi])
        temp.set_ylim([-1, 1])

    plt.savefig(f'{TOP_DIR}/out/dataset_tree.png')

    X = pd.DataFrame([], columns=['x', 'y'])

    df = df.rename(columns={'x1':'x', 'y1':'y'})
    X = pd.concat([X, df[['x', 'y']]], axis=0)
    df = df.drop(['x', 'y'], axis=1)

    df.index = range(dim, dim * 2)
    df = df.rename(columns={'x2':'x', 'y2':'y'})
    X = pd.concat([X, df[['x', 'y']]], axis=0)
    df = df.drop(['x', 'y'], axis=1)

    Y = np.zeros(dim * 2)
    Y[dim:] = 1.

    test_xx = np.linspace(0, 2*np.pi, num=col)
    test_yy = np.linspace(-1, 1, num=row)
    x, y = np.meshgrid(test_xx, test_yy)

    #
    # test_x = (x0, y0), (x1, y0), ..., (xN, y0), (x0, y1), ..., (xN, yN)
    #
    test_x = np.hstack([x[0].reshape(col, 1), y[0].reshape(col, 1)])
    for i in range(1, row):
        temp = np.hstack([x[i].reshape(col, 1), y[i].reshape(col, 1)])
        test_x = np.vstack([test_x, temp])

    run_decision_tree(X.values, Y, test_x)
    run_random_forest(X.values, Y, test_x)
    run_extra_tree_classifier(X.values, Y, test_x)

    plt.savefig(f'{TOP_DIR}/out/dataset_binary_classification_tree.png')
    plt.clf()
