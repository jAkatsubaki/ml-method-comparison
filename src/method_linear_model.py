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

def run_svm(X, Y, mesh_data):
    print("""
--------------------------------

        Support Vector Machine

--------------------------------""") 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    # LinearSVC
    linear_svc = LinearSVC()
    linear_svc.fit(X, Y)

    # 予測　
    Y_pred = linear_svc.predict(X_test)

    # 評価
    score = linear_svc.score(X_test, Y_test)

    coef = linear_svc.coef_[0]
    intercept = linear_svc.intercept_

    line = np.linspace(-1, np.pi * 2)
    coef = linear_svc.coef_[0]
    intercept = linear_svc.intercept_

    print("score = %.3f" % (score))
    print("Coef =", coef)
    print("Intercept =", intercept)

    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(linear_svc.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[3] , linewidth=0, s=2)
    fig.axes[3].set_title('SVM')

def run_logistic_regression(X, Y, mesh_data):
    print("""
--------------------------------

        Logistic Regression

--------------------------------""")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    # LogisticRegression
    logreg = LogisticRegression(penalty='l2', solver="sag")
    logreg.fit(X, Y)

    # 予測　
    Y_pred = logreg.predict(X_test)

    #
    # 評価
    #
    # 平均絶対誤差(MAE)
    mae = mean_absolute_error(Y_test, Y_pred)
    # 平方根平均二乗誤差（RMSE）
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # スコア
    score = logreg.score(X_test, Y_test)

    line = np.linspace(-1, np.pi * 2)
    coef = logreg.coef_[0]
    intercept = logreg.intercept_

    print("MAE = %.3f,  RMSE = %.3f,  score = %.3f" % (mae, rmse, score))
    print("Coef =", coef)
    print("Intercept =", intercept)

    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(logreg.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[2] , linewidth=0, s=2)
    fig.axes[2].set_title('Logistic Regression')

def run_perceptron(X, Y, mesh_data):
    print("""
--------------------------------

        Perceptron

--------------------------------""")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    # Perceptron
    ppt = Perceptron()
    ppt.fit(X_train, Y_train)

    # 予測
    Y_pred = ppt.predict(X_test)

    # 平均絶対誤差(MAE)
    mae = mean_absolute_error(Y_test, Y_pred)
    # 平方根平均二乗誤差（RMSE）
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # スコア
    score = ppt.score(X_test, Y_test)

    line = np.linspace(-1, np.pi * 2)
    coef = ppt.coef_[0]
    intercept = ppt.intercept_

    print("MAE = %.3f,  RMSE = %.3f,  score = %.3f" % (mae, rmse, score))
    print("Coef =", coef)
    print("Intercept =", intercept)

    df = pd.DataFrame(mesh_data, columns=["x", "y"])
    df["val"] = pd.Series(ppt.predict(mesh_data)).apply(lambda x: "#e0d3cc" if x==0. else "#a87f5d")
    sns.scatterplot(df["x"], df["y"], c=df["val"], ax=fig.axes[1] , linewidth=0, s=2)
    fig.axes[1].set_title('Perceptron')

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
    
    plt.savefig(f'{TOP_DIR}/out/dataset_linearmodel.png')

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

    run_perceptron(X.values, Y, test_x)
    run_logistic_regression(X.values, Y, test_x)
    run_svm(X.values, Y, test_x)

    plt.savefig(f'{TOP_DIR}/out/dataset_binary_classification_linearmodel.png')
    plt.clf()
