from numpy import *
import operator

def creatDataSet():
    group = array([[1.0 , 1.1],[1.0 , 1.0],[0 , 0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
# group,labels = creatDataSet()

#print(group,'\n',lables)

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]                           #shape[0]就是读取矩阵第一维度的长度
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                      #将一个矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()                # 从小到大排序，返回的是index
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1  #记录每个类的个数
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#print (classify0([0,0.5],group,labels,3))

def file2matrix(filename):

    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index=index+1
    return returnMat,classLabelVector

# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#print(datingDataMat,datingLabels)

# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# #print(datingDataMat[:,0])
#
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels)) #用标签来区分大小颜色
#
# plt.show()
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
# normMat,ranges,minVals = autoNorm(datingDataMat)

# print(normMat)
# print(ranges)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with :%d,the real answer is : %d"%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount +=1.0
    print("the total error rate is : %f"%(errorCount/float(numTestVecs)))
# datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTals = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of icr cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTals,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult - 1])
#classifyPerson()

def img2vector(filename):
    returnVect = zeros(1,1024)
    fr = open(filename)
    for i in range (32):
        lineStr = fr.readline()
        for j in range (32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
img2vector(r'C:\lcf\coding\machinlearning\2.3\trainingDigits\0_12.txt')
