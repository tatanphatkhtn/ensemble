import numpy  as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
bagging = BaggingClassifier(base_estimator = None, n_estimators = 5, bootstrap = False, bootstrap_features = True)

class Framework:

	def __init__(self, number_of_iter, algorithm = bagging, confident = None, verbose = False):

		if(confident != None): 
			self.__confident = confident
			self.__is_ensemble = 1
		else:
			self.__is_ensemble = 0

		self.__number_of_iter = number_of_iter
		self.extended_lal_data = None
		self.extended_data_target = None
		self.__algorithm = algorithm
		self.remained_unl = None
		self.__verbose = verbose
		self.duplicate = float(0)
		self.label_names = None
		self.labelSum = {}
	def __print_wrapper(self, msg, level = None):
		if(self.__verbose == 'QUIET'): return 0
		if((self.__verbose == 'STANDARD' and level == None) or \
			(self.__verbose == 'DETAIL' and level == 1) or \
			(self.__verbose == 'DETAIL' and level == None)):
			print msg

	def __fit_with_ensemble(self, lal_data, unl_data, data_target, train_size):

		vprint = self.__print_wrapper

		number_of_iter = self.__number_of_iter

		vprint('')
		vprint('-----------Framework detail------------')
		vprint('-Unlalel data: ' + str(len(unl_data)))
		vprint('-Lalbel data: ' + str(len(lal_data)))
		vprint('-Number of feature: ' + str(lal_data.shape[1]))
		vprint('-Learning by ensemble method')
		vprint('-Number of loop: ' + str(number_of_iter))
		vprint('-Training size: ' + str(train_size))
		vprint(self.__algorithm)
		vprint('-----------------------------------')
		vprint('')

		random_slice = np.random.choice(a = unl_data.shape[0],  size  = train_size, replace = False)
		unl_data_trained = unl_data[random_slice, :] #sample without replacement
		unl_data = np.delete(unl_data, random_slice, axis = 0)

		bagging = self.__algorithm

		bagging.fit(lal_data, data_target) #Init ensemble

		for i in range(0, number_of_iter):
			if(len(unl_data_trained) == 0): break

			predicted_result = bagging.predict_proba(unl_data_trained)
			predicted_target = bagging.predict(unl_data_trained)

			#major voting for a class need to exceed confident -> abs(col[1] - col[2]) > conf  
			predicted_result_mask = abs(predicted_result[:,1] - predicted_result[:,0]) > self.__confident 

			predicted_unl_data = unl_data_trained[predicted_result_mask, :]
			predicted_target = predicted_target[predicted_result_mask]


			#form new dataset
			lal_data = np.concatenate((lal_data, predicted_unl_data))
			data_target = np.concatenate((data_target, predicted_target))
			vprint( "Round  " + str(i) + ": " + str(len(predicted_unl_data)) + " guessed",1)

			vprint(str(self.label_names[0]) + ": "  + str(len(predicted_target[predicted_target == self.label_names[0]])),1)
			vprint(str(self.label_names[1]) + ": "  + str(len(predicted_target[predicted_target == self.label_names[1]])),1)
			unl_data_trained =  unl_data_trained[~predicted_result_mask, :] #unl = unl - predicted


			random_slice = np.random.choice(a = unl_data.shape[0],  size  = train_size - len(unl_data_trained), replace = False)
			more_trained_unl_data = unl_data[random_slice, :]
			unl_data = np.delete(unl_data, random_slice, axis = 0)		


			unl_data_trained = np.concatenate((unl_data_trained, more_trained_unl_data))
			# print 'unl_data_trained: ' + str(len(unl_data_trained))

			bagging.fit(lal_data, data_target)
			# print bagging.oob_score_
			
			intersec = np.prod(np.transpose(bagging.estimators_samples_), axis = 1)
			# print intersec
			# print np.bincount(intersec)
			# print len(np.bincount(intersec))

			if len(np.bincount(intersec)) == 2:
				self.duplicate = 1.0 * np.bincount(intersec)[1] / len(bagging.estimators_samples_[0])						
			else:				
				self.duplicate = 0
			# print self.duplicate

			vprint ('-------------------',1)
		vprint('')
		vprint('---------------Result----------------')
		vprint ('Unlabel remained: ' + str(len(unl_data)))
		self.remained_unl = unl_data
		self.extended_lal_data = lal_data
		self.extended_data_target = data_target
		vprint ('Lalbel Extended: ' + str(len(self.extended_lal_data)))
		bincount = np.bincount(self.extended_data_target == self.label_names[0])
		vprint(str(self.label_names[0]) + ": "  + str(bincount[1]))
		vprint(str(self.label_names[1]) + ": " + str(bincount[0]))

		self.labelSum[self.label_names[0]] = bincount[1]
		self.labelSum[self.label_names[1]] = bincount[0]


	def __fit_without_ensemble(self, lal_data, unl_data, data_target, train_size):

		number_of_iter = self.__number_of_iter
		unl_data_trained = unl_data[np.random.choice(a = unl_data.shape[0],  size  = train_size , replace = False), :] #sample without replacement
		algorithm = self.__algorithm

		algorithm.fit(lal_data, data_target)

		for i in range(0, number_of_iter):
			if(len(unl_data_trained) == 0): break

			predicted_unl_data = unl_data_trained
			predicted_target = algorithm.predict(unl_data_trained)


			#form new dataset
			lal_data = np.concatenate((lal_data, predicted_unl_data))
			data_target = np.concatenate((data_target, predicted_target))


			unl_data_trained = unl_data[np.random.choice(a = unl_data.shape[0],  size  = train_size , replace = False), :]

			algorithm.fit(lal_data, data_target)	

		self.extended_lal_data = lal_data
		self.extended_data_target = data_target


	def save_to_file(self):
		np.save('./ensemble-result/lal_data',self.extended_lal_data)
		np.save('./ensemble-result/target',self.extended_data_target)

	def fit(self, lal_data, unl_data, data_target, train_size):	
		
		self.label_names = np.unique(data_target)
		self.labelSum = {
			self.label_names[0]: 0,
			self.label_names[1]: 0
		}
		if(self.__is_ensemble):
			self.__fit_with_ensemble(lal_data, unl_data, data_target, train_size)
		else: 
			self.__fit_without_ensemble(lal_data, unl_data, data_target, train_size)
	
	def evaluate_model(self, fold):
		print 'Testing with ' + str(fold) + ' folds'
		print "Number of labled data predicted: " + str(len(self.extended_lal_data))

		print str(self.label_names[0]) + ": "  + str(len(self.extended_data_target[self.extended_data_target == self.label_names[0]]))
		print str(self.label_names[1]) + ": "  + str(len(self.extended_data_target[self.extended_data_target == self.label_names[1]]))

		#test accuracy
		print 'Testing '
		cv = KFold(n_splits=fold, random_state=0)
		# cv =LeaveOneOut()	
		cv_result = cross_val_score(self.__algorithm , self.extended_lal_data, self.extended_data_target, cv=cv)
		print "avg accuracy: " +  str(cv_result.mean())
		print "detail : " + str(cv_result)

