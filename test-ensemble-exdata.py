from ensemble import Framework 
from sffs import sffs
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import datetime

import matplotlib.pyplot as plt


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='En-SemiFS')
parser.add_argument('-l', help='Label  2d matrix path', dest='lal_path', action='append', required=True)
parser.add_argument('-u', help='Unlabel 2d matrix path', dest='unl_path', action='append', required=True)
parser.add_argument('-t', help='Label array path (like Scikit target array)', dest='tar_path', action='append', required = True)
parser.add_argument('-a', help='Attribute array (Ex: [ABC,BC,DA]) path', dest='attr_path', action='append', required = True)
parser.add_argument('-v', help='Log mode: QUIET, STANDARD, DEFAULT', dest='verbose', action='append')


args = parser.parse_args()

lal_path  =  args.lal_path[0]
unl_path  =  args.unl_path[0]
tar_path  = args.tar_path[0]
attr_path =  args.attr_path[0]
verbose = "STANDARD"
if(args.verbose): verbose = args.verbose

print args.verbose
print lal_path
print unl_path
print tar_path
print attr_path

 
lal_data = np.load(lal_path)
unl_data = np.load(unl_path)
target = np.load(tar_path)
attr = np.load(attr_path)


clf = tree.DecisionTreeClassifier()
# output = open('run-' + str(datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")) + '.txt', 'w')

# # random data
random_slice = np.random.choice(a = lal_data.shape[0],  size  =  80* len(lal_data)/100, replace = False)

train_set = lal_data[random_slice,:]
train_target = target[random_slice]

test_set = np.delete(lal_data, random_slice, axis = 0)
test_target = np.delete(target, random_slice)

# # print 'Before Semi'
# # clf.fit(train_set, train_target)
# # print clf.score(test_set, test_target)

# # sffs_model = sffs(0.0000000000000000000000000000001)

# # sffs_model.loaddatafromarray(lal_data, target , attr)
# # accuracy, selected_feature =  sffs_model.sffs_predefine(clf, 50, 30, 5)

# # train_set_filtered = train_set[:, selected_feature]
# # test_set_filtered = test_set[:, selected_feature]

# print 'Before Semi'
# output.write('Before Semi\n')
sffs_model = sffs(0.0000000000000000000000000000001)

SemiAccuracyArr = []
SFFSAccuracyArr = []
NumOfEstimatorArr = []
DuplicateRatioArr = []


clf.fit(train_set, train_target)
beforeSemi = clf.score(test_set, test_target)

sffs_model.loaddatafromarray(train_set, train_target, attr)
accuracy, selected_feature =  sffs_model.sffs_predefine(clf, 50, 70, 5)

SemiAccuracyArr.append(beforeSemi)
NumOfEstimatorArr.append(0)
DuplicateRatioArr.append(0)
SFFSAccuracyArr.append(accuracy)

# output.write(str(beforeSemi) + '\n')

for num_of_estimators in range(2,30):

	# print len(train_set)
	bagging = BaggingClassifier(base_estimator = None, n_estimators = num_of_estimators, bootstrap = True)
	fw = Framework(number_of_iter = 10 , algorithm = bagging, confident =  0.5,  verbose =  verbose)

	fw.fit(train_set, unl_data, train_target, train_size = 250)

	# print 'After Semi'
	clf.fit(fw.extended_lal_data, fw.extended_data_target)
	afterSemi = clf.score(test_set, test_target)


	SemiAccuracyArr.append(afterSemi)
	NumOfEstimatorArr.append(num_of_estimators)
	DuplicateRatioArr.append(fw.duplicate)
	
	sffs_model.loaddatafromarray(fw.extended_lal_data, fw.extended_data_target , attr)
	accuracy, selected_feature =  sffs_model.sffs_predefine(clf, 50, 70, 5)
	SFFSAccuracyArr.append(accuracy)

	# output.write(str(afterSemi) + ' ')
	# print afterSemi
	# print fw.duplicate
	# print fw.labelSum
	# output.write(str(fw.duplicate) + ' ')
	# output.write(str(fw.labelSum) + ' ')

plt.subplot(211)
plt.plot(NumOfEstimatorArr, SemiAccuracyArr)
plt.xlabel('Number Of Estimators')
plt.ylabel('Accuracy')

plt.subplot(211)
plt.plot(NumOfEstimatorArr, SFFSAccuracyArr , c = 'r')


plt.subplot(212)
plt.plot(DuplicateRatioArr, SFFSAccuracyArr)
plt.xlabel('Duplicate Ratio between classifiers')
plt.ylabel('Accuracy')
plt.show()


# 	print '----------------------------------------------'

# output.close()
# # print 'After SFFS'
# # sffs_model.loaddatafromarray(fw.extended_lal_data, fw.extended_data_target , attr)
# # accuracy, selected_feature = sffs_model.sffs_predefine(clf, 50, 30, 5)

# # train_set_filtered = train_set[:, selected_feature]
# # test_set_filtered = test_set[:, selected_feature]

# # clf.fit(train_set_filtered, train_target)
# # print clf.score(test_set_filtered, test_target)
# # print attr[selected_feature]








