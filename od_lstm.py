#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM


class lstm:
    def __init__(self,filename1,filename2):
        cg0 = pd.read_excel(filename1)
        cg1 = pd.read_excel(filename2)
        
        cg0_asc = cg0[(cg0.호선 == 8) & (cg0.구분 == '상선')] 
        cg0_desc = cg0[(cg0.호선 == 8) & (cg0.구분 == '하선')]
        cg1_asc = cg1[(cg1.호선 == 8) & (cg1.구분 == '상선')] 
        cg1_desc = cg1[(cg1.호선 == 8) & (cg1.구분 == '하선')]
        
        # 혼잡도 정보 제외한 열 drop
        cg0_asc0 = cg0_asc.drop(['조사일자', '호선', '역번호', '역명','구분'],axis=1)
        cg1_asc0 = cg1_asc.drop(['조사일자', '호선', '역번호', '역명','구분'],axis=1)

        cg0_asc0 = cg0_asc0.drop(cg0_asc0.index[0])
        cg1_asc0 = cg1_asc0.drop(cg1_asc0.index[0])
        
        sum1=pd.DataFrame(cg0_asc0.sum())
        cg0_asc_sum = sum1.transpose()
        self.cg0_asc_sum_list=cg0_asc_sum.values/16
        self.cg0_asc_sum_list=self.cg0_asc_sum_list.tolist()

        sum2=pd.DataFrame(cg1_asc0.sum())
        cg1_asc_sum = sum2.transpose()
        self.cg1_asc_sum_list=cg1_asc_sum.values/16
        self.cg1_asc_sum_list=self.cg1_asc_sum_list.tolist()
        
    
    def make_dataset(self, data, window_size):
        data_list = []
        target_list = []

        for i in range(len(data) - window_size) :
            data_list.append(np.array(data[i:i+window_size]))
            target_list.append(np.array(data[i+window_size]))

        return np.array(data_list), np.array(target_list)
    
    def make_data(self):
        x_data = np.array(self.cg0_asc_sum_list[0])
        test_data = np.array(self.cg1_asc_sum_list[0])
        
        self.train_x, self.train_y = self.make_dataset(x_data, 10)
        self.train_x = np.reshape(self.train_x, (self.train_x.shape[0], self.train_x.shape[1], 1))
        self.train_y = np.reshape(self.train_y, (self.train_y.shape[0], 1, 1))
        
        self.test_x, self.test_y = self.make_dataset(test_data, 10)
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], self.test_x.shape[1], 1))
        
    def learn(self):
        self.make_data()
        
        model = Sequential()
        model.add(LSTM(100, input_shape = (self.train_x.shape[1],1)))
        model.add(Dense(1))
        
        model.compile(optimizer='adam',loss='mean_squared_error')
        print("혼잡도 예측 시작...", file=sys.stdout)
        model.fit(self.train_x, self.train_y, epochs=500, batch_size=12, verbose=0)
        
        self.pred = model.predict(self.test_x)
        
        self.pred = self.pred.reshape(1,134).tolist()[0]


