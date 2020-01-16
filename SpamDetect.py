import time
from pandas import read_csv
from pandas import set_option
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# import dataset
filename = 'spambase.data.csv'
dataset = read_csv(filename, header=None)

# seperate dataset
array = dataset.values
X = array[:, 0:57]
Y = array[:, 57]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# models:LR,NB,CART
models = {}
models['LDA'] = LinearDiscriminantAnalysis()
models['CART'] = DecisionTreeClassifier()
models['LR'] = LogisticRegression()
# models['KNN'] = KNeighborsClassifier()
# models['SVM'] = SVC()
# models['NB'] = GaussianNB()


# stratified ten-fold cross-validation tests
num_folds = 10
results = []
for key in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold)
    results.append(cv_results)
    print(cv_results)
    print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))

start = time.clock()
model1 = LinearDiscriminantAnalysis()
model1.fit(X=X_train, y=Y_train)
predictions1 = model1.predict(X_validation)
end = time.clock()
print(accuracy_score(Y_validation,predictions1))
print('Running time--LDA: %s seconds'% (end-start))
print classification_report(Y_validation, predictions1)

start2 = time.clock()
model2 = LogisticRegression()
model2.fit(X=X_train, y=Y_train)
predictions2 = model2.predict(X_validation)
end2 = time.clock()
print(accuracy_score(Y_validation,predictions2))
print('Running time--LR: %s seconds'% (end2-start2))
print classification_report(Y_validation, predictions2)


start3 = time.clock()
model3 = DecisionTreeClassifier()
model3.fit(X=X_train, y=Y_train)
predictions3 = model3.predict(X_validation)
end3 = time.clock()
print(accuracy_score(Y_validation,predictions3))
print('Running time--CART: %s seconds'% (end3-start3))
print classification_report(Y_validation, predictions3)
