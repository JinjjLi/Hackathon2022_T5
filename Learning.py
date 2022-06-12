import DataIO as io
import numpy as np
import Models as cm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from datetime import datetime
from keras.models import load_model
import Plot as plot
import time


FilePath = "C:/Data/SeiswareJun/"
FileName = "Form_table.csv"
modelPath = "C:/Data/SeiswareJun/model.h5"


learningColumns = [5,6,7,8]
targetColumn = 9
r_test = 0.8       # divide ratio of test dataset    20%   use for evaluation 0: no test
r_val = 0.2        # divide ratio of validation dataset 20%, use for training 0:no validation


def evaluation(model, x, y):
    result = model.evaluate(x, y)
    y_predict = model.predict(x)
    print('Evaluation: ', result)


def training_model(LearningD, targetD):
    x_train = LearningD[:int(len(LearningD) * r_test),:]
    y_train = targetD[:int(len(targetD) * r_test), :]
    x_test = LearningD[int(len(LearningD) * r_test):,:]
    y_test = targetD[int(len(targetD) * r_test):, :]
    print(len(LearningD[0]), len(LearningD), len(x_train),len(x_test) )
    model = cm.create_model_BP_A(len(LearningD[0]))
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=r_val)

    model.save(modelPath)
    plot.plot_results2(history)
    if len(x_test) > 0:
        evaluation(model, x_test, y_test)


if __name__ == '__main__':
    db = io.Database()
    WellN, FormN, TVDSS, X, Y, LearningData,targetData = db.readData(FilePath + FileName, learningColumns, targetColumn)
    formations = set(FormN)

    LearningD = db.normalization(np.array(LearningData).reshape(-1,len(learningColumns)).astype(np.float64),
                                 FilePath + "stat.csv", True)
    targetD = db.normalizationT(np.array(targetData).reshape(-1,1).astype(np.float64), FilePath + "stat.csv" )

    training_model(LearningD, targetD)

    str_resultRidgeCV, Y_predictRidgeCV = cm.RidgeCVModel(FilePath, LearningD, targetD)
    print(str_resultRidgeCV)

    str_resultSGD, Y_predictSGD = cm.SGDRegressorModel(FilePath, LearningD, targetD)
    print(str_resultSGD)
    print("Done")
