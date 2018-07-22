import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
svm_clf = svm.SVC()
from sklearn.model_selection import LeaveOneOut
from sklearn import  tree

from sklearn.model_selection import KFold

def testWithSffs(alg, train_set, train_target, test_set, test_target, attr_mask):
	train_set = train_set[:, attr_mask]
	test_set = test_set[:, attr_mask]
	print "-----------------Test---------------"
	print "Model:"
	print alg
	print "Number of feature: " + str(len(attr_mask))
	print train_set.shape
	print test_set.shape
	print '------------'

	alg.fit(train_set, train_target)
	result = alg.score(test_set,test_target)
	print "Result: " + str(result)



cv1 = KFold(n_splits= 15)
cv2 = LeaveOneOut()
id3 = tree.DecisionTreeClassifier()
lal_data = np.load('./pp-result/filtered/filteredLalMatrix0.5.npy')
target = np.load('./ng-result/target.txt')
selectedFeatures = np.load('./selectedFeatures.npy')



train_set = lal_data[:, selectedFeatures]
# test_set = target[:, selectedFeatures]
cv_result = cross_val_score(svm_clf , lal_data, target, cv=cv2)
print "avg accuracy: " +  str(cv_result.mean())
print "detail : " + str(cv_result)

cv_result = cross_val_score(svm_clf , train_set, target, cv=cv2)
print "avg accuracy: " +  str(cv_result.mean())
print "detail : " + str(cv_result)

