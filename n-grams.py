import numpy
import pickle #built-in file handling module from python
import sys, getopt  #handling commandline params
import os #delete temp file, see saveToFile function for more


#use numpy only
#data file with format, "sequence label"

class glbClass: #singleton class, contains global properties , just for encapsulation
    # Here will be the instance stored.
    __instance = None
    __currentDatasetIndx = 0
    __resultDir = './ng-result/'
    __dataOffset = [0,0,0] #number of data [unlabeled, svr, non-svr]

    data = numpy.array([]) #contain data read from file, type array of strings
    target = numpy.array([],  dtype = str) # array of labels

    sparseMatrix = [] 

    numberOfAttr = 0
    listOfAttr = {} #type: dictionary ,template: {attrName1: index,attrName2: index + 1 , ...}, attribute collection from extracting whole dataset
    # reversedListOfAttr = {}
    opts = getopt.getopt(sys.argv[1:],"u:s:n:") # u: unlabeled data file, s: svr data file, n: non-svr data file

    fileList = numpy.array(["./data/unlabeled.txt", "./data/svr.txt", "./data/non-svr.txt"], dtype = "S256") #read from agr if agr provided, default values for otherwise
    dataFileNames = numpy.array(["unlabeled-data.pkl", "svr-data.pkl", "non-svr-data.pkl"], dtype = "S256") #save finished data to file with default names
    targetFileNames = numpy.array(["unlabeled-target.pkl", "svr-target.pkl", "non-svr-target.pkl"], dtype = "S256") #save to finished target data to file with default names


    def init(self):
        for opt, arg in self.opts[0]:
            if opt  == "-u": #unlabeled data
                print "unlabeled dataset: " + arg + '\n'
                self.fileList[0] = arg
            elif opt == "-s": #svr data
                print "svr dataset: " + arg + '\n'
                self.fileList[1] = arg
            elif opt == "-n": #non-svr data
                print "non-svr dataset: " + arg + '\n'
                self.fileList[2] = arg

    #doing with local attr
    def appendAttr(self, attrName):
        self.listOfAttr[attrName] = self.numberOfAttr
        self.numberOfAttr = self.numberOfAttr + 1
    def getAttrIndex(self, attrName):
        return self.listOfAttr[attrName]
    def appendTarget(self, label):
        self.target = numpy.append(self.target, label)

    #handling file
    # def writeDownRecord(self, record):
    #     tempFile = open(self.__resultDir + 'temp', 'a')
    #     record = record.tolist()
    #     pickle.dump(record, tempFile) #write record down to file
    #     tempFile.close() #apply change to file and clear mem
    def saveRecord(self, record):
        self.sparseMatrix.append(record)
        return

    def saveToFile(self):
        print "Len: " + str(len(self.sparseMatrix))
        outFile = open("./" + self.__resultDir + "matrix-unl", 'w')
        print "Saving matrix..."
        for i in range(0, len(self.sparseMatrix)):
            print str(i)
            standardizedRecord = numpy.pad(self.sparseMatrix[i], (0, self.numberOfAttr - len(self.sparseMatrix[i])), 'constant')
            numpy.save(outFile, standardizedRecord)
            if (i == self.__dataOffset[0] - 1):
                print "Change to label file"
                outFile.close()
                outFile = open("./" + self.__resultDir + "matrix-lal", 'w')
        outFile.close()

        print "Saving attr..."
        attrListFile = open('./ng-result/attr.pkl', 'w')
        pickle.dump(self.listOfAttr, attrListFile)
        attrListFile.close()
        outFile = open('./ng-result/target.txt', 'w')
        numpy.save( outFile, self.target)
        outFile.close()
            # print '---->Saved attr to ./result/attr.pkl'    
            # os.remove(self.__resultDir + './temp') #clean out the temp file which occupy lot of space
            # print '---->Remove temp file'


    def fetchDataSet(self):
        print '### Current dataset: ' + str(self.fileList[self.__currentDatasetIndx])
        self.data = numpy.genfromtxt(fname = self.fileList[self.__currentDatasetIndx], delimiter = ' ', dtype = (str,str))
        if(self.__currentDatasetIndx is 0):
            self.__dataOffset[self.__currentDatasetIndx] = len(self.data)
        else:
            self.__dataOffset[self.__currentDatasetIndx] =  self.__dataOffset[self.__currentDatasetIndx - 1] + len(self.data)
        print self.__dataOffset
        self.__currentDatasetIndx = self.__currentDatasetIndx + 1



    #---------------singleton implement------------ 
    @staticmethod
    def getInstance():
        """ Static access method. """
        if glbClass.__instance == None:
            glbClass()
        return glbClass.__instance 

    def __init__(self):
        """ Virtually private constructor. """
        if glbClass.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            # self.__middleFile.close() 
            os.mkdir('./ng-result')
            glbClass.__instance = self
    #-----------------------------------------------


def slideWindow(sequence, window):
    extractSeq = sequence[window[0]:window[1]]
    window[0]+=1
    window[1]+=1
    return extractSeq

def ngram(ngramInstance, sequence, n):
    window = [0, n]
    glbObj = glbClass.getInstance()
    while(window[0] <= len(sequence) - n):
        extractSeq = slideWindow(sequence, window)
        if(extractSeq not in glbObj.listOfAttr): #case 1 : new attr found!
            glbObj.appendAttr(extractSeq)
            ngramInstance = numpy.append(ngramInstance, 1)
        else: #case 2: hit from old attr
            attrIndex = glbObj.getAttrIndex(extractSeq)
            ngramInstance[attrIndex] = ngramInstance[attrIndex] + 1
    # print ngramInstance
    return ngramInstance
def loopNgram(dataTuple):
    print 'Start parsing...'
    sequence = dataTuple[0]
    label = dataTuple[1]
    glbObj = glbClass.getInstance()
    previousNumberOfAttr = glbObj.numberOfAttr #for monitoring logs purpose 

    ngramInstance = numpy.zeros((glbObj.numberOfAttr), dtype=int) #init ngramInstance with already-created attributes
    
    for i in range(2,41):
        ngramInstance = ngram(ngramInstance, sequence, i) 

    print '->Finish parsing!'
    print '+Current number of attr: ' + str(glbObj.numberOfAttr)
    print 'Increased: ' + str(glbObj.numberOfAttr - previousNumberOfAttr) 
    print '----------------------------------------'
    glbObj.saveRecord(ngramInstance)
    if(label != "NULL"):
        glbObj.appendTarget(label) #write down label
        print label

glbObj = glbClass.getInstance()

def analyze(dataset):
    print 'Total: ' + str(len (dataset))
    print '---------------------------'
    # print glbObj.data
    for index in range(0, len(dataset)):
        # print glbObj.data[index]
        print '-Current tuple index: ' + str(index)
        loopNgram(dataset[index])
    # glbObj.saveToFile()




glbObj.init()
print glbObj.fileList

glbObj.fetchDataSet()
analyze(glbObj.data)
glbObj.fetchDataSet()
analyze(glbObj.data)
glbObj.fetchDataSet()
analyze(glbObj.data)
glbObj.saveToFile()

print len(glbObj.listOfAttr)