from sklearn import datasets
from sklearn import tree
import numpy as np
import sys

class sffs(object):

    def loaddatafromfile(self, pathdataFile, pathtargetFile, pathattriFile, pathtestFile, pathtesttargetFile):
        dataFile = open(pathdataFile, 'rb')
        self.dataset = np.load(dataFile)

        targetFile = open(pathtargetFile, 'rb')
        self.target = np.load(targetFile)

        attriFile = open(pathattriFile, 'rb')
        self.attribute = np.load(attriFile)

        testFile = open(pathtestFile, 'rb')
        self.testset = np.load(testFile)

        testtargerFile = open(pathtesttargetFile, 'rb')
        self.testtarget= np.load(testtargerFile)

        dataFile.close()
        targetFile.close()
        attriFile.close()
        testFile.close()
        testtargerFile.close()

        # print "Num of feature: ", len(self.attribute)
        # print "Num of labeled: ", len(self.target)
        # print

    def loaddatafromarray(self, dataset, target, attribute, testset, testtarget):        
        self.dataset = dataset        
        self.target = target        
        self.attribute = attribute
        self.testset = testset
        self.testtarget = testtarget

        # print "Num of feature: ", len(self.attribute)
        # print "Num of labeled: ", len(self.target)        
        # print

    ##########################_SFFS_####################################
    def sffs_predefine(self, classifier, predefine):            
        accuracy = float(0)             
        stop_sffs = False
        selected_feature = np.array([], dtype='int64') # order features are selected
        extracted_feature = np.array([], dtype='int64')    
        total_feature = len(self.attribute) # total of features
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
                
                trainset_experiment = self.dataset[:,selected_feature_experiment]
                testset_experiment = self.testset[:,selected_feature_experiment]

                # caculate score when adding feature                
                classifier = classifier.fit(trainset_experiment, self.target)                               
                score = classifier.score(testset_experiment, self.testtarget)
            
                # print "Adding feature ", index_feature, " - Score is ", score
                if maximize_score <= score:
                    maximize_score = score
                    index_selected = index_feature

            # ADD FEATURE
            if index_selected != -1:
                # get feature to add
                selected_feature = np.append(selected_feature, index_selected)                
                accuracy = maximize_score

                # print "Num of features: ", len(selected_feature)
                # print selected_feature              
                # print "Accuracy: ", accuracy
            else:
                print "All remaining features make declined accuracy!"
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

                    for index_feature in range(0, len(selected_feature)):                                 
                        selected_feature_experiment = np.delete(selected_feature, [index_feature])  
                        # print selected_feature
                        # print selected_feature_experiment          

                        trainset_experiment = self.dataset[:, selected_feature_experiment]
                        testset_experiment = self.testset[:, selected_feature_experiment]

                        # caculate score when adding feature 
                        classifier = classifier.fit(trainset_experiment, self.target)
                        score = classifier.score(testset_experiment, self.testtarget)                                               
                       
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

            #             print "Extracted featue ", extract_element                                      
            #             print "Num of features: ", len(selected_feature)
            #             print selected_feature                                        
            #             print "Accuracy: ", accuracy
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
        print self.attribute[selected_feature]
        print
        return accuracy, selected_feature

    ##########################_SFFS_####################################