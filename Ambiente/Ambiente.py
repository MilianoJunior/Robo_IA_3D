import pandas as pd
import math
from Ambiente.Trade import Trade


class ambiente():
    def __init__(self,entrada_rnn,entrada_trader,name="Trade"):
        self.name = name
        self.contador = 0
        self.cont = 0
        self.entrada_rnn = entrada_rnn
        self.entrada_trader = entrada_trader
        self.media = pd.DataFrame(columns=['ganho']) 
        self.trader = Trade()
        self.metal = False
        self.actionA = 0
        self.A =[0,0]
    def sigmoid(self,x):
        return (x-44)/55

    def trade(self,action):
        done = False
        stop = -300
        gain = 300
        compra,venda,neg,ficha,comprado,vendido,recompensa = self.trader.agente(self.entrada_trader.values[self.cont],self.A[self.contador],stop,gain,0)
        self.contador += 1
        self.cont += 1
        self.A.append(action)
        recompensa = self.sigmoid(recompensa)
        # print('------------------')
        # print('contador: ',self.cont)
        # print('acao: ',action)
        # print('vendido: ',vendido)
        # print('comprado: ',comprado)
        # print('recompensa: ',recompensa)
        if comprado or vendido:
            self.metal = True
        if self.metal and (comprado == False and vendido == False):
            self.metal = False 
        if self.cont >= (len(self.entrada_rnn)-10):
            self.cont =0
        if self.contador >= 540:
            self.contador = 0
            self.A =[0,0]
            done = True
            self.media = self.media.append({'ganho': sum(neg.ganhofinal)},ignore_index=True)
            rolling = self.media.mean()
            print('ganho atual: ',sum(neg.ganhofinal),'N operacoes: ',len(neg.ganhofinal),' media: ',rolling.values[-1])
            self.trader.reset()
        return self.entrada_rnn.values[self.cont],recompensa,done
    def reset(self):
        self.contador = 0