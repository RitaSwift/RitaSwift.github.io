## 这是《机器学习实战》中的第一个算法：k-邻近算法

我将我所有的笔记都和代码整合到一起，以后关于这本书的笔记都是这样的

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

#!/usr/bin/env python    # -*- coding: utf-8 -*

"""
关于这个文档的主要功能和注意点
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.,1.1],[1.,1.],[0.,0.],[0.,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# k-近邻算法
"""
参数说明：
inX 用于分类的输入向量
dataSet 数据样本集
labels 标签集
k kNN算法中的k，最接近项目的规定数目
"""
def classify0(inX, dataSet, labels, k):
    #1. 距离计算
    #shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数
    #shape[a,b] a:返回行数，b：返回列数
    dataSetSize = dataSet.shape[0]
    #numpy.tile(A,B)函数 重复A元素，B次 B可以是int，int时行不重复，B是二维数组时，B[x,y] 行重复x次，列重复y次
    #测试的向量和数据集中的向量相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #a**2 ==>求a的平方 a**3 ==>求a的立方
    #算出距离的平方
    sqDiffMat = diffMat**2
    #sum函数解释 详见：https://blog.csdn.net/rifengxxc/article/details/75008427
    #axis=a 就是这个标志的位置是变的 其他位置是不变的和
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #将distances中的元素从小到大排列，并输出对应的索引
    sortedDistIndicies = distances.argsort()
    #声明一个字典
    classCount={}
    #从0-2,
    for i in range(k):
        #获取该索引对应的训练样本的标签
        ## index = sortedDistIndicies[i]是第i个最相近的样本下标
        voteIlabel = labels[sortedDistIndicies[i]]
        #classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        #然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #把分类结果进行排序，然后返回得票数最多的分类结果
        """sorted()函数的用法
        sort 与 sorted 区别：
        sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
        reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）"""
        #参照http://www.runoob.com/python/python-func-sorted.html
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#测试代码
# 这段代码没有看 是摘抄的测试代码 2018/9/17 20:58
if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)

# 使用k-邻近算法改进约会网站匹配结果 2018/9/17 20:58

#函数功能：将输入的文件中的字符串变成训练样本矩阵和类标签向量 2018/9/18 上午
def file2matrix(filename):
    #打开文件，文件类对象为fr
    fr = open(filename)
    #返回的是文件的内容，每一行以\n结尾,返回的类型是list
    arrayOLines = fr.readlines()
    #返回文件的行数
    numberOFLines = len(arrayOLines)
    #zeros该函数功能是创建给定类型的矩阵，并初始化为0，zeros((),float或者int等，默认是float)
    returnMat = zeros((numberOFLines,3))
    classLabelVector = []
    index = 0
    #依次访问每一行
    for line in arrayOLines:
        #去掉换行符，用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        line = line.strip()
        #像Java中的split一样，以制表符"\t"来拆分单元，返回一个list
        listFromLine = line.split("\t")
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #将喜欢还是不喜欢放在list classLabelVector
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#上述约会的测试代码
if  __name__ == "__main__":
    #打开的文件名
    filename = "datingTestSet2.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    print(datingDataMat)
    print(datingLabels)

#使用Matplotlib创建散点图 2018/9/18 上午
"""import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()"""

#改变显示的图形大小和颜色 2018/9/18 上午
"""ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
           15.0 * array(datingLabels), 15.0  * array(datingLabels))"""

#函数功能：将数值归一化 2018/9/18 上午
def autoNorm(dataSet):
    """
    min(0)返回该矩阵中每一列的最小值

    min(1)返回该矩阵中每一行的最小值

    max(0)返回该矩阵中每一列的最大值

    max(1)返回该矩阵中每一行的最大值
    """
    #求出每列的最小值
    minVals = dataSet.min(0)
    #求出每列的最大值
    maxVals = dataSet.max(0)
    #求出每列的范围
    ranges = maxVals - minVals
    #生成一个和dataSet同规模的并且值都是0的矩阵
    normDataSet = zeros(shape(dataSet))
    #取得矩阵的行数
    m = dataSet.shape[0]
    #tile是复制函数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet, ranges, minVals


#利用分类器代码测试约会网站的正确率 2018/9/18 上午
def datingClassTest():
    #用来测试的数据占原数据的比例
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minvals = autoNorm(datingDataMat)
    #获取矩阵的行数
    m = normMat.shape[0]
    #测试的数据为100行
    numTestVecs = int(m*hoRatio)
    #计算出错的次数
    errorCount = 0.0
    #从0-100行来检测分类器的正确率
    for i in range(numTestVecs):
        #得到分类器判断的结果
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], 3)
        #打印分类器的结果和正确的结果做对比
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        #如果两个结果不相等，就把错误的数目加一
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


#整个约会网站的预测函数 2018/9/18 上午
def classifyPerson():
    #关于预测可能结果的list集合
    resultList = ["not at all","in small doses","in large doses"]
    #打游戏所占时间比
    #raw_input()是python内部函数，但是在python3.X进行了整合 都整合到input（）函数中
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    #获取原数据集中的矩阵和结果集
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    #数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #将结果变成一个数据集
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult - 1])

#关于重载模块
"""
import importlib
importlib.reload(kNN)
"""
#2018/9/18 上午，先自己想
def img2vector(filename):
    """
    该函数创建1*1024的NumPy数组，然后打开给定的文件，循环读出文件的前32行，并且
    将每行的头32个字符值存储在NumPy数组中，然后返回数组
    发现文件中都是32*32的
    :param filename: 文件的名字
    :return:
    """
    #自己坑坑洼洼写的代码 2018/9/18 下午，先自己想,自己关于读出文件前32行就不会写了，看看再自己写的
    #声明一个数组，1*1024的数组
    returnVect = zeros((1,1024))
    #打开文件
    fr = open(filename)
    #按照行遍历
    for i in range(32):
        #读取一行的数据
        lineStr = fr.readline()
        #读取一行中的每一列的数据
        for j in range(32):
            #将数据装入到向量
            returnVect[0,32*i+j] = int(lineStr[j])
    #返回完整的1*1024的向量
    return returnVect

#手写数字识别系统的测试代码 2018/9/18 晚上
#listdir函数可以列出给定目录的文件名
from os import listdir
def handwritingClassTest():
    #保存相对应的文件所代表的数字
    hwLabels = []
    #提取这个目录下的所有文件名
    trainingFileList = listdir("trainingDigits")
    #获取所有的文件总数
    m = len(trainingFileList)
    #将所有的数据存放在trainingMat矩阵中，每一行代表一个文件的所有数据
    trainingMat = zeros((m,1024))
    for i in range(m):
        #去掉文件格式的后缀
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        #取出这个手写图形的真正代表的数字  2018/9/18 18:57 感觉不用谢classNumStr 写成classNum就好啦 因为后面已经强转成int型
        #2018/9/19 上午 需要转成int，后面判断测试的数字的时候需要用到
        classNumStr = int(fileStr.split("_")[0])
        #将其加入集合内
        hwLabels.append(classNumStr)
        #将对应的文件数据传入到相应的矩阵行中
        trainingMat[i,:] = img2vector("trainingDigits/%s" % fileNameStr)
    #得到所有用来测试的文件名
    testFileList = listdir("testDigits")
    #统计分类器出错的情况
    errorCount = 0
    #统计测试的文件数目
    mTest = len(testFileList)
    #遍历所有的文件，将统计分类器判断的结果 2018/9/19 上午
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
        print("the total number of errors is: %d" % errorCount)
        print("the total error rate is: %f" % (errorCount/float(mTest)))
        """2018/9/19 上午 最后的这个函数handwritingClassTest()执行效率不高，DOS窗口运行好长时间，所以效率不高  
        因为为每个测试向量所计算的距离就有2000次，每个距离向量的测试还包括1024个维度浮点计算，此外还需要为测试向量准备2MB的存储空间
        K-决策树就是k-邻近算法的优化版，减少存储空间和计算时间的消耗
        """









- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RitaSwift/RitaSwift.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
