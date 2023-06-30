# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:08:27 2023

@author: kjath
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, batch_size):
    	# super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()
        self.batch_size = batch_size
        
        # batch_size = 100
        self.layer = nn.Sequential(
            
            # [7,1,256,256] -> [7,16,252,252]
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            
            # [7,16,252,252] -> [7,32,248,248]
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            
            # [7,32,248,248] -> [7,32,124,124]*9
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            
            # [7,32,124,124] -> [7,64,120,120]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            
            # [7,64,120,120] -> [7,64,60,60]
            nn.MaxPool2d(kernel_size=2,stride=2)          
        )
        self.fc_layer = nn.Sequential(
        	
            # [7,64*62*62] -> [1,100]
            nn.Linear(64*60*60,100),                                              
            nn.ReLU(),
            # [1,100] -> [1,3]
            nn.Linear(100,3)
            # nn.Softmax(dim=1) # 넣으니까 정확도 확 떨어짐                                
        )
        
    def forward(self,x):
    	# self.layer에 정의한 연산 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100,나머지]로 변환
        out = out.view(self.batch_size,-1)
        # self.fc_layer 정의한 연산 수행    
        out = self.fc_layer(out)
        return out