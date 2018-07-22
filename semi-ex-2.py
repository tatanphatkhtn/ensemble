import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
svm_clf = svm.SVC()
from sklearn.model_selection import LeaveOneOut
from sklearn import  tree

from sklearn.model_selection import KFold
cv1 = KFold(n_splits= 5)
cv2 = LeaveOneOut()
id3 = tree.DecisionTreeClassifier()
lal_data = np.load('./pp-result/filtered/filteredLalMatrix0.5.npy')
target = np.load('./ng-result/target.txt')
selectedFeatures = np.load('./selectedFeatures.npy')

cv_result = cross_val_score(svm_clf , lal_data, target, cv=cv1)
print "avg accuracy: " +  str(cv_result.mean())
print "detail : " + str(cv_result)

