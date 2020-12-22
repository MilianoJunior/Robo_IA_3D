# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:25:56 2020

@author: jrmfi
"""
import chardet
import pandas as pd

class Data():
    def __init__(self,num_days):
        self.num_days = num_days
    def import_data(self):
        with open('dados/M4.csv', 'rb') as f:
            result = chardet.detect(f.read())  # or readline if the file is large
    
        base = pd.read_csv('dados/M4.csv', encoding=result['encoding'])
        entrada_rnn,entrada_trader = self.training_assess(base,self.num_days)
        return entrada_rnn,entrada_trader
        
    def duration(self,base):
        index = 0
        for i in base.values:
            base1 = i[0].split(':')
            base.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
            index += 1
        return base
    def training_assess(self,base,num_days = 565,colunas = ['Hora','dif', 'retacao +',
                                                            'retracao -', 'RSI', 'M22M44', 
                                                            'M22M66', 'M66M44', 'ADX', 'ATR',
                                                            'Momentum', 'Force']):
        colunas1 = ['Hora', 'open', 'high', 'low', 'close'] 
        entrada_RNN = pd.DataFrame(data=base[-num_days:-1].values,columns=base.columns)      
        entrada_trade = pd.DataFrame(data=base[-num_days:-1].values,columns=base.columns)
        entrada_RNN = entrada_RNN.drop(['Data'], axis=1)
        # entrada_RNN = entrada_RNN[colunas]
        entrada_trade = entrada_trade[colunas1]
        entrada_RNN = self.duration(entrada_RNN)
        train_mean = entrada_RNN.mean(axis=0)
        train_std = entrada_RNN.std(axis=0)
        entrada_RNN = (entrada_RNN - train_mean) / train_std
        return entrada_RNN,entrada_trade