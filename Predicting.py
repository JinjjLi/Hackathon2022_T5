import numpy as np
import DataIO as io
import tensorflow as tf


FilePath = "C:/Data/SeiswareJun/"
FileName = "Form_table.csv"
modelPath = "C:/Data/SeiswareJun/model.h5"

learningColumns = [5,6,7,8]
targetColumn = 9


if __name__ == '__main__':
    db = io.Database()

    WellN, FormN, TVDSS, X, Y, LearningData, targetData = db.readData(FilePath + FileName,
                                                                      learningColumns, targetColumn)
    LearningD = db.normalization(np.array(LearningData).reshape(-1, len(learningColumns)).astype(np.float64),
                                 FilePath + "stat.csv",False)
    model = tf.keras.models.load_model(modelPath)
    predictData = model.predict(LearningD)
    # print(predictData)
    result = db.convertPredict(predictData, len(learningColumns))

    finalR = list()
    for i in range(len(WellN)):
        finalR.append(WellN[i] + ',')
        finalR.append(FormN[i] + ',')
        finalR.append(TVDSS[i]+ ',')
        finalR.append(X[i] + ',')
        finalR.append(Y[i]+ ',')
        for j in range(len(learningColumns)):
            finalR.append(LearningData[i* len(learningColumns) + j ] + ',')
        finalR.append(result[i] + '\n')

    db.save_file(FilePath + "result.csv", finalR, 'w')

    print("Predict done")