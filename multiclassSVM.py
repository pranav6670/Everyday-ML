from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)