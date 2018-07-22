import numpy as np
import sys
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

class sffs(object):

    def __init__(self):
        self.__classifiers_list = []
        clf_dt = tree.DecisionTreeClassifier()
        clf_svm = svm.SVC()  
        clf_kneighbors = KNeighborsClassifier(3)     

        self.add_classifier(clf_dt)
        self.add_classifier(clf_svm)      
        self.add_classifier(clf_kneighbors)
    
    def add_classifier(self, classifier):
        if classifier not in self.__classifiers_list:
            self.__classifiers_list.append(classifier)

    def calculate_score(self, train, train_target, test, test_target):
        score = float(0)        

        for classifier in self.__classifiers_list:
            classifier = classifier.fit(train, train_target)
            score += classifier.score(test, test_target)
      
        score /= len(self.__classifiers_list)
        return score


    def loaddatafromfile(self, pathdataFile, pathtargetFile, pathattriFile, pathtestFile, pathtesttargetFile):
        dataFile = open(pathdataFile, 'rb')
        self.__dataset = np.load(dataFile)

        targetFile = open(pathtargetFile, 'rb')
        self.__target = np.load(targetFile)

        attriFile = open(pathattriFile, 'rb')
        self.__attribute = np.load(attriFile)

        testFile = open(pathtestFile, 'rb')
        self.__testset = np.load(testFile)

        testtargerFile = open(pathtesttargetFile, 'rb')
        self.__testtarget= np.load(testtargerFile)

        dataFile.close()
        targetFile.close()
        attriFile.close()
        testFile.close()
        testtargerFile.close()

        print "Num of feature: ", len(self.__attribute)
        print "Num of labeled: ", len(self.__target)
        print

    def loaddatafromarray(self, dataset, target, attribute, testset, testtarget):        
        self.__dataset = dataset        
        self.__target = target        
        self.__attribute = attribute
        self.__testset = testset
        self.__testtarget = testtarget

        print "Num of feature: ", len(self.__attribute)
        print "Num of labeled: ", len(self.__target)        
        print

    ##########################_SFFS_####################################
    def sffs_predefine(self, predefine):            
        accuracy = float(0)             
        stop_sffs = False
        selected_feature = np.array([], dtype='int64') # order features are selected
        extracted_feature = np.array([], dtype='int64')    
        total_feature = len(self.__attribute) # total of features
        
        if (predefine > total_feature):
            print "Predefine is ", predefine, " but just having ", total_feature, "features"
            return accuracy, selected_feature
        
        print "Running SFFS..."
        print
        # loop until convergence the number of features
        while stop_sffs == False:
            # step 1: Inclusion
            # find feature which contributed to highest accuracy
            # print "INCLUSION"
            maximize_score = accuracy
            index_selected = -1         
            for index_feature in range(0, total_feature):                
                if index_feature in selected_feature or index_feature in extracted_feature:                    
                    continue               
                               
                selected_feature_experiment = np.append(selected_feature, index_feature)
                
                trainset_experiment = self.__dataset[:,selected_feature_experiment]
                testset_experiment = self.__testset[:,selected_feature_experiment]

                # caculate score when adding feature                
                score = self.calculate_score(trainset_experiment, self.__target, testset_experiment, self.__testtarget)
            
                print "Adding feature ", index_feature, " - Score is ", score
                if maximize_score <= score:
                    maximize_score = score
                    index_selected = index_feature

            # ADD FEATURE
            if index_selected != -1:
                # get feature to add
                selected_feature = np.append(selected_feature, index_selected)                
                accuracy = maximize_score
                print "Adding feature ", index_selected, " - Accuracy: ", accuracy
                # print "Num of features: ", len(selected_feature)
                # print selected_feature              
                # print "Accuracy: ", accuracy
            else:
                print "When adding the remaining features reduces the accuracy!"
                break

            # Step 2: Conditional Exclusion
            # extract feature in matrix
            if len(selected_feature) > 2:
                # print "EXCLUSION"
                stop_extract = False
                while stop_extract == False:                    
                    stop_extract = True
                    maximize_score = accuracy

                    if len(selected_feature) <= 2:
                        break
                    seleted_num = len(selected_feature)
                    for index_feature in range(0, seleted_num):                                 
                        selected_feature_experiment = np.delete(selected_feature, [index_feature])  
                        # print selected_feature
                        # print selected_feature_experiment          

                        trainset_experiment = self.__dataset[:, selected_feature_experiment]
                        testset_experiment = self.__testset[:, selected_feature_experiment]

                        # caculate score when adding feature 
                        score = self.calculate_score(trainset_experiment, self.__target, testset_experiment, self.__testtarget)                                           
                       
                        # print "Extracting feature ", selected_feature[index_feature], " - Score is ", score
                        if maximize_score < score:
                            index_extract = index_feature
                            maximize_score = score
                            stop_extract = False
                    
                    if stop_extract == False:                                              
                        accuracy = maximize_score
                        
                        extract_element = selected_feature[index_extract]
                        selected_feature = np.delete(selected_feature, [index_extract])
                        extracted_feature = np.append(extracted_feature, extract_element)

                        print "Extracted featue ", extract_element, " Accuracy: ", accuracy
            #             print "Num of features: ", len(selected_feature)
            #             print selected_feature                                        
            #             print
            # print

            # stop_sffs condition
            if len(selected_feature) == predefine:
                stop_sffs = True                        
        print               
        print "FINAL RESULT"
        print "Accuracy: ", accuracy
        print "Num of selected features: ", len(selected_feature)
        print selected_feature
        print self.__attribute[selected_feature]
        print
        return accuracy, selected_feature

    ##########################_SFFS_####################################