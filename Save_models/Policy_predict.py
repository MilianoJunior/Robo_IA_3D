
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import chardet
import pandas as pd
from comunica import  Comunica

num_days = 5400
# array(['2020.12.10', '09:16', -45, 20, 15, 39.16, 32.52, 74.7, -42.19,
#        28.08, 113.57, 99.71, -145104.62, 6737.0, -97.93, -163.02, -83.02,
#        26.17, 18.59, -77.59, 168.76, 36.44, 113130.25, 113467.76,
#        112792.74], dtype=object)

with open('M4.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

base = pd.read_csv('M4.csv', encoding=result['encoding'])

def duration(base):
    index = 0
    for i in base.values:
        base1 = i[0].split(':')
        base.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
        index += 1
    return base

base1 = pd.DataFrame(data=base[-num_days:-1].values,columns=base.columns) 
entrada_RNN = pd.DataFrame(data=base[-num_days:-1].values,columns=base.columns)      
entrada_RNN = entrada_RNN.drop(['Data','open', 'high', 'low', 'close','OBV','Acumulacao'], axis=1)
entrada_RNN = duration(entrada_RNN)
media = entrada_RNN.mean(axis=0)
std = entrada_RNN.std(axis=0)
po = 0
def confere(p,entrada_RNN,po):
    if entrada_RNN.Hora.values[-340 + po] == p[0] :
        for i in range(len(entrada_RNN.values[-340])):
            if entrada_RNN.values[-340+ po][i] == p[i]:
                print('igual')
            else:
                print('diferente: ',entrada_RNN.values[-340+ po][i],p[i],entrada_RNN.values[-340+ po][i]-p[i])
        po += 1
        return po
    return po

new_model = tf.keras.models.load_model('modelo_01')

HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()

while True:
    p,addr = R.runServer(s)
    jm = (p-media)/std
    jm = np.array([jm])
    # print(jm.shape)
    # po = confere(p,entrada_RNN,po)
    state = tf.constant(jm, dtype=tf.float32)
    previsao1 = new_model.predict(state)
    previsao2 = np.argmax(previsao1[0][0])
    d3 = p[0]
    print('recebido: ',p[0])
    # print(previsao2)
    # print('----------------')
    # d3 = previsao1[0][0][previsao2]
    if previsao2 == 0:
        print('Sem operacao')
    if previsao2 == 1:
        flag = "compra-{}".format(d3)
        # flag ="compra"
        print('compra: ',previsao2)
        R.enviaDados(flag,s,addr)
    if previsao2 == 2:
        flag = "venda-{}".format(d3)
        # flag = "venda"
        print('venda: ',previsao2)
        R.enviaDados(flag,s,addr)


