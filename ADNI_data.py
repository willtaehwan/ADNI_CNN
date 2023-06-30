# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:03:32 2023

@author: kjath
"""
import pandas as pd
import nibabel as nib
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# import torch.nn as nn 

class ADNI_data:
    def __init__(self, csv, image):
        # csv : ADNI 정보를 담은 csv 파일 경로
        # image : MRI 사진 파일이 담긴 폴더 경로
        self.csv = csv
        self.image = image
        
        # ADNI 정보 csv 파일 load
        self.binfo = pd.read_csv(self.csv)
        
        # MRI 사진 파일의 상대 경로를 new_path라는 새로운 컬럼 추가
        self.binfo['new_path'] = [self.image + '/' + self.binfo.Path[i].split('/')[-1] for i in range(len(self.binfo))]
        
        # new_path에 저장된 경로를 통해 nibabel 라이브러리로 .gz파일을 읽어와 brain_list라는 리스트에 numpy형태로 저장
        self.brain_list = [nib.load(self.binfo['new_path'][i]).get_fdata() for i in range(len(self.binfo))]
        # Ex) 70개의 (256*256*256) 크기의 numpy array 저장
        
        # DX -> OneHotEncoding
        # self.binfo['DX_label'] = OneHotEncoder().fit_transform(self.binfo['DX_original'])
        self.binfo['DX_label'] = list(OneHotEncoder(sparse_output=False).fit_transform(self.binfo[['DX_original']]))

        
    def len_tensor(self): # data 길이
        return len(self.brain_list)
    
    def tensor(self): # 이미지 데이터 tensor화
        self.b_t = torch.tensor(np.array(self.brain_list))
        
        return self.b_t
    
    def binfo(self):
        return self.binfo
    
    def brain_list(self):
        return self.brain_list
    
    def brain_128(self): # 단면도에 대한 이미지만 추출, Tensor화하여 저장
        self.data_128 = [MinMaxScaler().fit_transform(self.brain_list[i][128]) for i in range(len(self.brain_list))]
        
        self.d_t = torch.tensor(np.array([[i] for i in self.data_128]))
        
        return self.d_t
        
    def label(self): # 라벨링 데이터 추출

        self.label = torch.tensor(self.binfo.DX_label)
        
        return self.label
        
        
        
        
        
        
        
        