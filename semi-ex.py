from ensemble import Framework 
import numpy as np
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from sffs import sffs
from sffs_score_multi_classifiers import sffs
from sklearn import  tree

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

id3 = tree.DecisionTreeClassifier()
cv = KFold(n_splits= 5, random_state=0)

gnb_clf = GaussianNB()
svm_clf = svm.SVC()
sgd_clf = SGDClassifier(loss="hinge", penalty="l2")

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


# bagging = BaggingClassifier(base_estimator = None, n_estimators = 5, bootstrap = True)
# fw = Framework(number_of_iter = 10 , algorithm = bagging, confident =  0.5,  verbose =  "STANDARD")

lal_data = np.load('./pp-result/filtered/filteredLalMatrix0.5.npy')
unl_data = np.load('./pp-result/filtered/filteredUnlMatrix0.5.npy')
target = np.load('./ng-result/target.txt')
attr = np.load('./pp-result/filtered/filteredAttr0.5.npy')

# random_slice = np.random.choice(a = lal_data.shape[0],  size  =  50* len(lal_data)/100, replace = False)

# train_set = lal_data[random_slice,:]
# train_target = target[random_slice]

# test_set = np.delete(lal_data, random_slice, axis = 0)
# test_target = np.delete(target, random_slice)


# print lal_data.shape
# print unl_data.shape
# print attr.shape
# print target.shape

# id3.fit(lal_data, target)
# beforeSemi = id3.score(test_set, test_target)
# print "Init accuracy: " +  str(beforeSemi)
# print "detail : " + str(cv_result)

# acc_arr = []
# num_es  = []
sffs_model = sffs()
# acc_arr.append(beforeSemi)
# num_es.append(0)






bagging = BaggingClassifier(base_estimator = None, n_estimators = 19, bootstrap = True)
fw = Framework(number_of_iter = 10 , algorithm = bagging, confident =  0.5,  verbose =  "DETAIL")
fw.fit(lal_data,unl_data,target)
# id3.fit(fw.extended_lal_data, fw.extended_data_target)
# afterSemi = id3.score(test_set, test_target)
# print "After semi accuracy: " +  str(afterSemi)

sffs_model.loaddatafromarray(fw.extended_lal_data, fw.extended_data_target , attr, lal_data, target)
accuracy, selected_feature =  sffs_model.sffs_predefine(25)

np.save('./selectedFeatures', selected_feature)

# print 'After sffs: ' + str(accuracy)

# testWithSffs(id3,fw.extended_lal_data, fw.extended_data_target,test_set,test_target,selected_feature)
# testWithSffs(svm_clf,fw.extended_lal_data, fw.extended_data_target,test_set,test_target,selected_feature)

# testWithSffs(sgd_clf,fw.extended_lal_data, fw.extended_data_target,test_set,test_target,selected_feature)

# testWithSffs(gnb_clf,fw.extended_lal_data, fw.extended_data_target,test_set,test_target,selected_feature)


# acc_arr.append(afterSemi)
# num_es.append(i)




# plt.plot(num_es, acc_arr)
# plt.xlabel('Number Of Estimators')
# plt.ylabel('Accuracy')

# plt.show()
