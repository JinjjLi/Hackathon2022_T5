import matplotlib.pyplot as plt
import numpy
from matplotlib import cm
import pandas as pd
import numpy as np
import DataIO as io
from matplotlib.ticker import LinearLocator


def plot_results(history):
    # show results
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.plot(history.history['mae'])
    # plt.plot(history.history['val_mae'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

def plot_results2(history):
    # show results with two pictures
    plt.figure()
    ax1 = plt.subplot2grid((1,2),(0,0))
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='upper right')
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.plot(history.history['mae'])
    ax2.plot(history.history['val_mae'])
    ax2.set_title('mae')
    ax2.set_ylabel('mae')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper right')

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("D:/Data/2RBinfo.csv")
    db = io.Database()

