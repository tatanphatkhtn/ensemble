# -----------build full matrix
import numpy as np
import pickle
import gc
import os
import sys, getopt

freq = 0.5
opts = getopt.getopt(sys.argv[1:],"f:") 
for opt, arg in opts[0]:
    freq = arg

class dataPreProcess:
    def __init__(self, lalPath, unlPath, attrPath, resultDir, freq):
        self.lalPath = lalPath
        self.unlPath = unlPath
        self.resultDir = resultDir
        self.attrPath = attrPath
        self.fullMatrix = None
        self.fullLalMatrix = None
        self.fullUnlMatrix = None
        self.freq = freq
        if(not os.path.exists(resultDir)): os.mkdir(resultDir)

    def __freeMatrixMem(self):
        self.fullMatrix = []
        self.fullLalMatrix = []
        self.fullUnlMatrix = []
        gc.collect()
        print 'Collected'


    def __buildFullMatrix(self):
        print '---------Build full matrix-------'
        dataset = []
        print 'Loading lalbel data'
        dataFile = open(self.lalPath, 'r')
        while(True):
            try:
                record = np.load(dataFile).tolist()
                dataset.append(record)
            except IOError:
                dataFile.close()
                break
        print 'Loading unlalbel data'
        dataFile = open(self.unlPath, 'r')
        print 'Building Full matrix'
        while(True):
            try:
                record = np.load(dataFile).tolist()
                dataset.append(record)
            except IOError:
                dataFile.close()
                break
        dataset = np.array(dataset)

        np.save(self.resultDir + 'fullMatrix', dataset)
        print 'Saved!'
        self.fullMatrix = dataset
        self.fullMatrixLength = len(dataset)
        print "Length: " + str(self.fullMatrixLength)
        print 'Saved!'

    # ---------Build fullLalMatrix
    def __buildFullLalMatrix(self):
        print '---------Build full lal matrix-------'
        
        dataset = []
        dataFile = open(self.lalPath, 'r')
        while(True):
        	try:
        		record = np.load(dataFile).tolist()
        		dataset.append(record)
        	except IOError:
        		dataFile.close()
        		break
        dataset = np.array(dataset)

        np.save(self.resultDir + 'fullLalMatrix', dataset)
        self.fullLalMatrix = dataset




    #---------Build fullUnLalMatrix
    def __buildFullUnlMatrix(self):
        print '---------Build full unl matrix-------'
        
        dataset = []
        dataFile = open(self.unlPath, 'r')
        while(True):
        	try:
        		record = np.load(dataFile).tolist()
        		dataset.append(record)
        	except IOError:
        		dataFile.close()
        		break
        dataset = np.array(dataset)

        np.save(self.resultDir  + 'fullUnlMatrix', dataset)
        self.fullUnlMatrix = dataset
        print 'Saved!'



    # ---------Calculate sum

    # import numpy as np

    def __calSum(self):
        print '---------Cal sum-------'

        # dataFile = open(self.resultDir + 'fullMatrix.npy', 'r')
        dataset = self.fullMatrix

        print dataset.shape
        sumArr = []
        for i in range(0, dataset.shape[1]):
        	sumArr.append(sum(dataset[:,i]))
        # sumArr =  np.array(sumArr)
        np.save(self.resultDir + 'sum', np.array(sumArr))
        print 'Finished'

    # ----------------------Reverse attr
    def __reverseAttr(self):
        print '---------Reverse attr-------'

        attrFile = open(self.attrPath, 'r')
        attrDict = pickle.load(attrFile)
        attrFile.close()
        attrList = [None]*len(attrDict)
        for key,value in attrDict.iteritems():
        	attrList[value] = key

        np.save( self.resultDir + 'reverseAttr', np.array(attrList))
        print 'Finished'





    # ----------Project Feature 
    # print 'Ready to use data info'
    def __dimReduceByFreq(self):
        print '---------Reduce dim-------'

        print 'Project Feature'
        sumArr = np.load(self.resultDir + 'sum.npy').astype(float)
        attrList = np.load(self.resultDir +  'reverseAttr.npy')
        print 'Loading full matrix...'
        lmtx = self.fullLalMatrix
        umtx = self.fullUnlMatrix

        sumPercent = sumArr / 4287
        newAttrList = []
        sumBin = sumPercent > float(self.freq)

        # print np.bincount(sumBin)


        # np.save('./result/attrMask', sumBin)
        # print 'Saved attrMask'
        os.mkdir(self.resultDir + 'filtered')
        np.save(self.resultDir + 'filtered/filteredLalMatrix' + str(self.freq), lmtx[:, sumBin])
        np.save(self.resultDir + 'filtered/filteredUnlMatrix' + str(self.freq), umtx[:, sumBin])

        print 'Saved filtered matrix'

        for i in range(0, len(sumBin)):
        	if(sumBin[i] == True):
        		newAttrList.append(attrList[i])
        np.save(self.resultDir + 'filtered/filteredAttr' + str(self.freq) ,np.array(newAttrList))

        print 'Lal path: ' + self.resultDir + 'filtered/filteredLalMatrix' + str(self.freq)
        print 'Unl path: ' + self.resultDir + 'filtered/filteredUnlMatrix' + str(self.freq)
        print 'Fin'

    def tdimReduceByFreq(self):
        print '---------Reduce dim-------'

        print 'Project Feature'
        sumArr = np.load(self.resultDir + 'sum.npy').astype(float)
        print sumArr
        sumPercent = sumArr / self.fullMatrixLength
        print sumPercent
        sumBin = sumPercent > float(self.freq)
        print self.freq
        print sumBin

        return
        attrList = np.load(self.resultDir +  'reverseAttr.npy')
        print 'Loading full matrix...'
        lmtx = np.load(self.resultDir + "fullLalMatrix.npy")
        umtx = np.load(self.resultDir + "fullUnlMatrix.npy")


        newAttrList = []



        # print np.bincount(sumBin)


        # np.save('./result/attrMask', sumBin)
        # print 'Saved attrMask'
        # os.mkdir(self.resultDir + 'filtered')
        np.save(self.resultDir + 'filtered/filteredLalMatrix' + str(self.freq), lmtx[:, sumBin])
        np.save(self.resultDir + 'filtered/filteredUnlMatrix' + str(self.freq), umtx[:, sumBin])

        print 'Saved filtered matrix'

        for i in range(0, len(sumBin)):
            if(sumBin[i] == True):
                newAttrList.append(attrList[i])
        np.save(self.resultDir + 'filtered/filteredAttr' + str(self.freq) ,np.array(newAttrList))

        print 'Lal path: ' + self.resultDir + 'filtered/filteredLalMatrix' + str(self.freq)
        print 'Unl path: ' + self.resultDir + 'filtered/filteredUnlMatrix' + str(self.freq)

        print 'Fin'

    def runProcess(self):
        self.__buildFullMatrix()
        self.__calSum()
        self.__freeMatrixMem()
        self.__buildFullLalMatrix()
        self.__buildFullUnlMatrix()
        self.__reverseAttr()

        self.__dimReduceByFreq()

dp = dataPreProcess("./ng-result/matrix-lal","./ng-result/matrix-unl", "./ng-result/attr.pkl" , './pp-result/', freq)
dp.runProcess()
# dp.tdimReduceByFreq()
# import numpy as np

# attr_list = np.load('./result/filtered/filteredAttr.npy')
# # attr_list = np.load('./result/reverseAttr.npy')


# # print attr_list
# print len(attr_list)
# def filter(features):
#     i = 0
#     result = []
#     for i in features:
#     	print i
#         status = True
#         for j in features:
#             if (i == j) and (len(i) == len(j)):
#                 continue
#             if len(i) < len(j):
#                 if i in j:
#                     print j
#                     status = False
#                     break

# 	print '----------------------------------'             
#         result.append(status)
#     return result

# result = filter(attr_list)  
# print np.bincount(result)







#statistic
# import numpy as np
# dataFile = open('./result/fullLalMatrix.npy', 'r')
# print 'Loading...'
# mtx = np.load(dataFile)	
# print 'Finished!'
# uniqued_mtx = np.unique(mtx, axis = 0)
# print mtx.shape
# print len(uniqued_mtx)

