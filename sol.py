from __future__ import print_function

import pickle
import numpy as np
# Machine Learning
import sys
from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees,GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()

# Load
X_ids, X, y = pickle.load(open('/home/hduser/Xy.pkl', 'rb'))
y = y.reshape(y.shape[0])
X_ids = np.array(X_ids)
X = np.array(X)

# Split learn and final with outcome indices
i_final = (y == -1)
i_learn = (y > -1)

#split train and test arrays
X_ids_final = X_ids[i_final]
X_ids_learn = X_ids[i_learn]
X_final = X[i_final, :]
X_learn = X[i_learn, :]
y_learn = y[i_learn]

# print(X_learn.shape[0])
# print(y_learn.shape[0])
# print y_learn.shape[0]
# print(X_learn.shape[1])

#merge train array with their outcomes
test_data = np.column_stack((y_learn, X_learn))
# print(test_data.tolist())
# print(test_data.shape)
# for i in range(0,4630):
#     # print(test_data[i])
#     row = ''
#     for j in range(0,151):
#         row = row + str(X_final[i][j]) + ','
#     print(row[:-1])
ext=0
def parsePoint(line):
    values= [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

def parsePoint2(line):
    values= [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

#train data load
train_data_new = sc.textFile('/home/hduser/dataset.txt')
parsedData = train_data_new.map(parsePoint)
#test data load
test_data_new = sc.textFile('/home/hduser/testfile.txt')
test_final = test_data_new.map(parsePoint2)

# Split train and test
X_train, X_test = parsedData.randomSplit([0.8,0.2])

#train the classifier
model=GradientBoostedTrees.trainClassifier(X_train,categoricalFeaturesInfo={},numIterations=10)
#20% of training data
predictions=model.predict(X_test.map(lambda x: x.features))
labelsAndPredictions1 = X_test.map(lambda p: p.label).zip(predictions)

#test data
predictions1=model.predict(test_final.map(lambda x: x.features))
y_final = test_final.map(lambda p: p.label).zip(predictions1)


er =labelsAndPredictions1.filter(lambda (v, p): v != p).count() / float(X_train.count())
acc = (1 - er)*100
print('===============================================================')
print(model.toDebugString())
print('===============================================================')
for i in y_final.collect():
    print(i)
print('===============================================================')
print('Accuracy = ' + str(acc) + '%')
print('===============================================================')
print('Learned over classification Gradient Boosted Trees model:')
print('===============================================================')
