import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDRegressor
from joblib import dump, load
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def create_model_BP_A(para_num):
    # this function is used for Back Propagation (BP) Neural Network Algorithm
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(para_num,)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # model.compile(optimizer=tf.optimizers.RMSprop(), loss=tf.losses.categorical_crossentropy, metrics=["accuracy"])
    return model


def RidgeCVModel(path, X_sample,Y_sample ):
    model = RidgeCV()
    model.fit(X_sample, Y_sample)

    str_joblib = path + "_model_RidgeCV.joblib"
    print(str_joblib)
    dump(model, str_joblib)

    Y_predict = model.predict(X_sample)
    error = mean_squared_error(Y_sample, Y_predict)
    cc = r2_score(Y_sample, Y_predict)

#    print("Error:", error)
#    print("Coef:", model.coef_, len(model.coef_[0]))
#    print("intercept:", model.intercept_, len(model.intercept_))
#    print("cc:", cc, error)

    str_result = "Wight: "
    for i in model.coef_[0]:
        str_result += str(format(i, '0.4f')) + " "
    str_result += "\nIntercept: " + str(format(model.intercept_[0], '0.4f')) + "\n"
    str_result += "Error: " + str(format(error, '0.4f')) + "\n"
    str_result += "cc: " + str(format(cc, '0.4f')) + "\n"
    print("fitRidgeCV Done")
    return str_result, Y_predict


def SGDRegressorModel(path, X_sample, Y_sample):
    model = SGDRegressor()
    model.fit(X_sample, Y_sample)
    str_joblib = path + "_model_SGD.joblib"

    dump(model, str_joblib)

    Y_predict = model.predict(X_sample)
    error = mean_squared_error(Y_sample, Y_predict)
    cc = r2_score(Y_sample, Y_predict)

    # print("Error:", error, model.coef_[0])
#    print("Coef:", model.coef_, len(model.coef_[0]))
#    print("intercept:", model.intercept_, len(model.intercept_))
#    print("cc:", cc, error)

    str_result = "Wight: "
    for i in model.coef_:
        str_result += str(format(i, '0.4f')) + " "
    str_result += "\nIntercept: " + str(format(model.intercept_[0], '0.4f')) + "\n"
    str_result += "Error: " + str(format(error, '0.4f')) + "\n"
    str_result += "cc: " + str(format(cc, '0.4f')) + "\n"

    print("fitSGD Done")
    return str_result, Y_predict

if __name__ == '__main__':
    pass