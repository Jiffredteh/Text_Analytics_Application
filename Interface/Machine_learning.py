from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from Config import util

def logistic_regression_model(X_train, y_train, X_test, y_test):
    LRmodel = LogisticRegression(solver='lbfgs',max_iter=250)
    util.machine_learning_model(LRmodel,X_train, y_train, X_test, y_test, "Logistic Regression Model")


def naive_bayes_model(X_train, y_train, X_test, y_test):
    NaiveBayes = MultinomialNB()
    util.machine_learning_model(NaiveBayes,X_train, y_train, X_test, y_test, "Multinomial Naive Bayes Classifier")
    
def SGD_Classifier_model(X_train, y_train, X_test, y_test):
    SGDClassifier_model = SGDClassifier()
    util.machine_learning_model(SGDClassifier_model,X_train, y_train, X_test, y_test, "SGD Classifier")
