import csv
import numpy as np
import math as math
# from random import random
# from random import seed
import pandas as pd

class Database():
    def __init__(self):
        self.data = np.array([])
        self.d_max = list()
        self.d_min = list()
        self.td_max = 0
        self.td_min = 0
        self.numPara = 0

    def readData(self, load_file, learningC, targetC):
        LearngData = list()
        listTemp = list()
        targetData = list()
        Well_Name = list()
        Form_Name = list()
        Form_top_TVDSS = list()
        UTMX = list()
        UTMY = list()

        with open(load_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if 'Well_Name' == row[0]:
                    continue
                if not row :  # check if there is empty line
                    continue
                Well_Name.append(row[0])
                Form_Name.append(row[1])
                Form_top_TVDSS.append(row[2])
                UTMX.append(row[3])
                UTMY.append(row[4])
                targetData.append(row[targetC])
                listTemp.clear()
                for i in learningC:
                    listTemp.append(row[i])

                LearngData.extend(listTemp)
        file.close()
        return Well_Name, Form_Name, Form_top_TVDSS, UTMX, UTMY,LearngData, targetData

    def save_fileData(self, file, data, ln, r='w'):
        f = open(file, r)
        n =0
        for row in data:
            if n == ln:
                f.write("\n")
                n = 0
            f.write(str(row) + ",")
            n += 1
        f.close()

    def save_file(self, file, data, r='w'):
        f = open(file, r)
        for row in data:
            f.write(row)
        f.close()

    def stat_max_min(self):
        for f in range(len(self.data[0])):
            self.d_max.append(float(np.max(self.data[:,f:f+1], axis=0)))
            self.d_min.append(float(np.min(self.data[:, f:f+1], axis=0)))
        # print(self.d_max)

    def load_max_min(self, path,ln):
        lines = list()
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            print(csv_reader)
            for row in csv_reader:
                lines.extend(row)
            for i in range(ln):
                self.d_max.append(float(lines[i]))
                self.d_min.append(float(lines[ln + 1 + i]))

    def __normalizing(self):
        for f in range(len(self.data[0])):
            m = self.d_max[f] - self.d_min[f]
            self.data[:, f:f + 1] = (self.data[:, f:f + 1] - self.d_min[f]) / m

    def normalization(self, testData, path, b_mark):
        # nor_type:True training data, false: testing data,
        self.data = testData
        self.d_max.clear()
        self.d_min.clear()
        if b_mark:
            self.stat_max_min( )
        else:
            print(testData[0])
            self.load_max_min(path, len(testData[0])+ 1)
        self.__normalizing()
        return self.data

    def normalizationT(self, testData, path):
        self.td_max= float(np.max(testData, axis=0))
        td_min= float(np.min(testData, axis=0))
        for i in range(len(testData)):
            testData[i] = (testData[i] - td_min)/ (self.td_max - td_min)

        self.d_max.append(self.td_max)
        self.d_min.append(td_min)

        arr = np.concatenate((np.array(self.d_max).astype(np.float64), np.array(self.d_min).astype(np.float64)))
        self.save_fileData(path, arr, len(self.d_max), 'w')

        return testData

    def convertPredict(self, predict, ln):
        datalist = list()
        print(self.d_min[ln], self.d_max[ln])
        for d in predict:
            strD = str(self.d_min[ln]+(self.d_max[ln] - self.d_min[ln])*d)
            strD = strD.replace('[','')
            strD = strD.replace(']', '')
            datalist.append(strD)
        return datalist
