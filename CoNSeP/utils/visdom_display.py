# -*- coding=utf-8 -*-

from visdom import Visdom
import numpy as np 

class Display(object):
  
    def __init__(self,envir,state,window_tag):
    
        self.viz = Visdom(env=envir)
        
        if state == 'TRAIN':
            self.train_display = TrainDisplay(self.viz,window_tag)
        elif state == 'TEST':
            pass
        else:
            raise Exception ("Error: Visdom state error!")
        
    def __call__(self,phase,X,Y):
       
        if phase == 'TRAIN':
            self.train_display(X,Y) # X：iter Y：Loss
        else:
            raise Exception ("Error: Visdom phase error!")

class TrainDisplay(object):

    def __init__(self, envir, window_tag):
        self.env = envir
        self.train_line = self.env.line(
                X = 0.1*np.ones(1),
                Y = 0.1*np.ones(1),
                opts = dict(
                    xlabel = 'Iteration',
                    ylabel = 'Loss',
                    title = ('Loss:'+window_tag),
                    )
                )

    def __call__(self,Iteration,Loss):

        Loss = 1 if Loss>1 else Loss
        self.env.line(
            X = np.array([Iteration]),
            Y = np.array([Loss]),
            win = self.train_line,
            update='append')    

class ValDisplay(object):

    def __init__(self, envir, window_tag):
        self.env = envir
        self.val_line = self.env.line(
                X = np.zeros(1),
                Y = np.zeros((1,4)),
                opts = dict(
                    xlabel = 'Epoch',
                    ylabel = 'Accuracy',
                    title = ('Accuracy:'+window_tag),
                    legend = ['AP','WP','NC','NLH'],
                    )
                )
    def __call__(self,Epoch,Precision_Dict):

        precision = list((Precision_Dict.values()))
        self.env.line(
            X = np.array([Epoch]),
            Y = np.array([precision]),
            win = self.val_line,
            update='append')
