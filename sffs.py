from sklearn import datasets
from sklearn import  tree
import numpy as np
import sys

class sffs(object):

    def __init__(self, eps):
        self.eps = eps       
    
    def loaddatafromfile(self, pathdataFile, pathtargetFile, pathattriFile):
        dataFile = open(pathdataFile, 'rb')
        self.dataset = np.load(dataFile)

        targetFile = open(pathtargetFile, 'rb')
        self.target = np.load(targetFile)

        attriFile = open(pathattriFile, 'rb')
        self.attribute = np.load(attriFile)

        dataFile.close()
        targetFile.close()
        attriFile.close()

        # print "Num of feature: ", len(self.attribute)
        # print "Num of labeled: ", len(self.target)
        # print

    def loaddatafromarray(self, dataset, target, attribute):        
        self.dataset = dataset        
        self.target = target        
        self.attribute = attribute

        # print "Num of feature: ", len(self.attribute)
        # print "Num of labeled: ", len(self.target)        
        # print
  
    def floatequal(self, numa, numb):        
        if numa == numb:
            return True
        
        if abs(numa - numb) <= self.eps:
            return True
        else:
            return False

    ##########################_SFFS_####################################
    def sffs_predefine(self, classifier, predefine, trainpercent, numiter):            
        accuracy = float(0)             
        stop_sffs = False
        selected_feature = [] # order features are selected
        extracted_feature = []
        matrix = np.array([])
        total_feature = len(self.attribute) # total of features
        
        print "Running SFFS..."
        print
        # loop until convergence the number of features
        while stop_sffs == False:            
            maximize_score = accuracy       
            
            # step 1: Inclusion
            # find feature which contributed to highest accuracy
            # print "INCLUSION"
            index_selected = -1          
            for index_feature in range(0, total_feature):                
                if index_feature in selected_feature or index_feature in extracted_feature:                    
                    continue
                
                temp_matrix = matrix # create temp_matrix to check
                feature = self.dataset[:,[index_feature]] # get feature to check          

                # add feature to temp_matrix
                if len(selected_feature) == 0:
                    temp_matrix = feature # when temp_matrix is empty
                else:
                    # when temp_matrix is not empty                     
                    temp_matrix = np.append(temp_matrix, feature, axis=1)
                
                # caculate score when adding feature
                score = 0
                for i in range(0, numiter):
                    random_slice = np.random.choice(a = temp_matrix.shape[0], size = trainpercent*len(temp_matrix)/100, replace = False)
                    
                    train_set = temp_matrix[random_slice,:]
                    train_target = self.target[random_slice]

                    test_set = np.delete(temp_matrix, random_slice, axis = 0)
                    test_target = np.delete(self.target, random_slice)

                    classifier = classifier.fit(train_set, train_target)

                    score += classifier.score(test_set, test_target)
                
                score /= numiter
 
                # print "Adding feature ", index_feature, " - Score is ", score
                if maximize_score < score:
                    maximize_score = score
                    index_selected = index_feature

            # ADD FEATURE
            if index_selected != -1:
                # get feature to add
                feature = self.dataset[:,[index_selected]]

                # add feature to matrix
                if len(selected_feature) == 0:
                    matrix = feature # when matrix is empty
                else:
                    # when matrix is not empty
                    matrix = np.append(matrix, feature, axis=1)
                
                selected_feature.append(index_selected)                
                accuracy = maximize_score        

                # print "Added features: " , index_selected
                # print "Num of features: ", len(selected_feature)
                # print selected_feature              
                # print "Accuracy: ", accuracy
            else:
                # print "Accuracy is not improved!"
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
                        temp_matrix = matrix
                        temp_matrix = np.delete(temp_matrix, [index_feature], axis=1)

                        score = 0
                        for i in range(0, numiter):
                            random_slice = np.random.choice(a = temp_matrix.shape[0], size = trainpercent*len(temp_matrix)/100, replace = False)

                            train_set = temp_matrix[random_slice,:]
                            train_target = self.target[random_slice]

                            test_set = np.delete(temp_matrix, random_slice, axis = 0)
                            test_target = np.delete(self.target, random_slice)

                            classifier = classifier.fit(train_set, train_target)
                            score += classifier.score(test_set, test_target)
                        
                        score /= numiter
                       
                        # print "Extracting feature ", selected_feature[index_feature], " - Score is ", score
                        if maximize_score < score:
                            index_extract = index_feature
                            maximize_score = score
                            stop_extract = False
                    
                    if stop_extract == False:                                              
                        accuracy = maximize_score

                        matrix = np.delete(matrix, [index_extract], axis=1)
                        extract_element = selected_feature[index_extract]
                        selected_feature.remove(extract_element)
                        extracted_feature.append(extract_element)

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