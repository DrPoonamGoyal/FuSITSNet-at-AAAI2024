#!/usr/bin/env python
# coding: utf-8
 

import numpy as np


import pandas as pd
import re
import os
import numpy as np
import time 

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import math


from torch.autograd import Variable
from torch.optim import *

from einops import rearrange
from torch import einsum

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()

 
crop = 'corn'
mdl = 'FuSITSNet'

 
test_years = [2020]
tst_year = test_years[0]
start_year = 2014
train_years = list (range(start_year,tst_year))
val_years = [tst_year-1]

print('train_years',train_years)
print('val_years',val_years)
print('test_years',test_years)
len(train_years), len(test_years), tst_year, crop, mdl

 
if crop == 'soy':
    timesteps_l =list (range(6,19)) 
    timesteps_m =list (range(16,43))
    
else:
    timesteps_l =list (range(5,20)) 
    timesteps_m =list (range(8,43)) 
 

len(timesteps_m), len(timesteps_l)

 
# GPU related info
cuda = 1
device = torch.device("cuda:1" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

 
rc_l = torchvision.transforms.Resize((2240, 2880),interpolation=T.InterpolationMode.NEAREST) 
rc_m = torchvision.transforms.Resize((150, 250),interpolation=T.InterpolationMode.NEAREST)  ##original 150,250

 
def unique(list1):
    unique_list = []
 
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

 
def process_state(dir_list):
    states = []
    counties = []
    
    for i in range(len(dir_list)):
        file = dir_list[i]
        locations=file[:-4].split("_")
        state = int(locations[0])
        states.append(state)
        stat = unique(states)

    return stat
    
    

 
def process_county(dir_list, state):
    states = []
    counties = []
    
    for i in range(len(dir_list)):
        file = dir_list[i]
        locations=file[:-4].split("_")
        state_fl = int(locations[0])

        if (state_fl != state):
            continue
        elif (state_fl == state):
            county_fl = int(locations[1])            
        counties.append(county_fl)
        county = unique(counties)
    return county
    
    

 
def common(a,b): 
    c = [value for value in a if value in b] 
    return c

 
landsat_folder='.../path to folder for landsat npz files'
modis_folder='.../path to folder for modis npz files'

print(landsat_folder)
print(modis_folder)

 
dir_list_ls1 = os.listdir(landsat_folder)
dir_list_md1 = os.listdir(modis_folder)

len(dir_list_ls1), len(dir_list_md1)

 
dir_list_ls2 = dir_list_ls1 
dir_list_md2 = dir_list_md1 

len(dir_list_ls2), len(dir_list_md2)

 
def getLandsatName(state, county,year):
    return str(state)+"_"+str(county)+"_"+str(year)+".npz"

def getModisName(state, county):
    return str(state)+"_"+str(county)+".npz"


states_ls = process_state(dir_list_ls2)
states_md = process_state(dir_list_md2)

states = common(states_ls,states_md)

dir_list_md = []
dir_list_ls = []
ls_train = []
ls_val = []
ls_test = []

tststates = []
tstcounties = []
print('state', states)

for state in states:
    counties_ls = process_county(dir_list_ls2,state)
    counties_md = process_county(dir_list_md2,state)
    counties = common(counties_ls,counties_md)

    tstcounties = tstcounties+counties
    state1 = np.array([state])
    state2 = state1.tolist()
    stt = state2*len(counties)
    tststates = tststates+stt
    for county in counties:
        m_file =getModisName(state, county)
        dir_list_md.append(m_file)
        for year in range(start_year,(tst_year+1)):
            l_file =getLandsatName(state, county,year)
            dir_list_ls.append(l_file)
            if year in train_years:
                ls_train.append(l_file)
            if year in val_years:
                ls_val.append(l_file)
            if year == tst_year:
                ls_test.append(l_file)             
print(len(dir_list_md),  len(dir_list_ls),len(ls_train),len(ls_val),len(ls_test))
            
len(tststates),len(tstcounties)           

 
class CustomImageDataset(Dataset):
    def __init__(self, dir_path, file_names, startyear,no_samples, ls_path, ls_list):
        self.img_dir = dir_path
        self.img_list = file_names
        self.no_samples = no_samples 
        self.startyear = startyear
        self.ls_path    = ls_path
        self.ls_list    = ls_list

    def __len__(self):
        return self.no_samples*len(self.img_list)

    def __getitem__(self, idx):

        file_num_md = idx//self.no_samples
    
        file_name = self.img_list[file_num_md]
        sample_num = idx%self.no_samples + self.startyear

        img_path = os.path.join(self.img_dir, file_name)
        files = np.load(img_path, allow_pickle=True)
        states   = torch.tensor(files[files.files[0]])
        counties = torch.tensor(files[files.files[1]])
        years    = torch.tensor(files[files.files[2]])
        red_img  = torch.tensor(files[files.files[3]])
        blue_img = torch.tensor(files[files.files[4]])
        green_img= torch.tensor(files[files.files[5]])
        nir_img  = torch.tensor(files[files.files[6]])
        swir_img = torch.tensor(files[files.files[7]])
        yld_img  = torch.tensor(files[files.files[8]])
 
        

        img_state  = states[0].numpy()
        img_county = counties[0].numpy()
        img_state  = np.asarray(img_state, dtype = 'int')
        img_county = np.asarray(img_county, dtype = 'int')
        

        red_img   = red_img[:,timesteps_m,:,:]
        blue_img  = blue_img[:,timesteps_m,:,:]
        green_img = green_img[:,timesteps_m,:,:]
        nir_img   = nir_img[:,timesteps_m,:,:]
        swir_img  = swir_img[:,timesteps_m,:,:]



        states = states[sample_num]
        counties = counties[sample_num]
        red_img   = red_img[sample_num]
        blue_img  = blue_img[sample_num]
        green_img = green_img[sample_num]
        nir_img   = nir_img[sample_num]
        swir_img  = swir_img[sample_num]
        years    = years[sample_num]

        states   = torch.unsqueeze(states, 0)   
        counties = torch.unsqueeze(counties, 0)
        years    = torch.unsqueeze(years, 0)        

        yld_img  = torch.unsqueeze(yld_img[0], 0) 
       
   
        md_img = torch.stack((red_img, blue_img, green_img,nir_img,swir_img), 0)
        md_img = md_img/255
        md_img = np.array(md_img, dtype=np.float16)
        md_img = torch.tensor(md_img)            
            

## ***************************** lnt *************************        
        file_num_ls = idx

        ls_file    = self.ls_list[file_num_ls]        

        ls_img_path    = os.path.join(self.ls_path, ls_file) 
        
        ls_f = np.load(ls_img_path, allow_pickle=True) 
        
        state_l1     = torch.tensor(ls_f['output_state'])
        county_l1     = torch.tensor(ls_f['output_county'])
        year_l1     = torch.tensor(ls_f['output_year'])
        
        red_l     = torch.tensor(ls_f['output_red'])
        blue_l     = torch.tensor(ls_f['output_green'])
        green_l     = torch.tensor(ls_f['output_blue'])
        nir_l     = torch.tensor(ls_f['output_nir'])
        swir_l     = torch.tensor(ls_f['output_swir'])       
        yld_l    = torch.tensor(ls_f['output_yld'])    
        
        red_l   = red_l[timesteps_l,:,:]
        blue_l  = blue_l[timesteps_l,:,:]
        green_l = green_l[timesteps_l,:,:]
        nir_l   = nir_l[timesteps_l,:,:]
        swir_l  = swir_l[timesteps_l,:,:]   
        
        
        state_l   = torch.unsqueeze(state_l1[0], 0)   
        county_l = torch.unsqueeze(county_l1[0], 0)       
        year_l = torch.unsqueeze(year_l1[0], 0)
        yld_l  = torch.unsqueeze(yld_l[0], 0) 
       
        
        ls_img = torch.stack((red_l, blue_l, green_l,nir_l,swir_l), 0)
        ls_img = ls_img/255
        ls_img    = np.array(ls_img, dtype=np.float16)
        ls_img    = torch.tensor(ls_img) 
        
# *********************rc ***********************

        ls_img    = rc_l(ls_img)
        md_img    = rc_m(md_img)    

        return_value =  [md_img,ls_img,states,counties,years,yld_l]           
        return return_value

 
dir_path = modis_folder 
dir_list = dir_list_md

start_idx = 0
dataset = CustomImageDataset(dir_path,dir_list,start_idx,len(train_years),landsat_folder,ls_train)


fnames = ['md_img','ls_img', 'states','counties','years', 'yld'] 
item   = dataset[20]
print([(fnames[i],item[i].shape) for i in range(len(item))])

 
start_idx = 0
vlidx = start_idx+len(train_years)-1
tstidx = start_idx+len(train_years)

train_data = CustomImageDataset(dir_path,dir_list,start_idx,len(train_years),landsat_folder,ls_train)
val_data = CustomImageDataset(dir_path,dir_list,vlidx,1,landsat_folder,ls_val)
test_data = CustomImageDataset(dir_path,dir_list,tstidx,1,landsat_folder,ls_test)

 
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

 
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    scores = F.softmax(scores, dim=-1)
    

        
    output = torch.matmul(scores, v)
    return output

 
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        
        k = k.to(device)
        q = q.to(device)
        v = v.to(device)        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
  

        mm = torch.matmul(q,k.transpose(2,3))


        mm1 = nn.Softmax(dim=1)

        sf = mm1(mm/math.sqrt(d_model))


        mm2 = torch.matmul(sf,v)

        scores = attention(q, k, v, self.d_k)


        concat = scores.transpose(1,2).contiguous()        .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

 

class cnn3d(nn.Module):


    def __init__(self, landsat_inc,sl,ec_dp,kernel_size):
        super(cnn3d, self).__init__() 
        self.conv1 = nn.Conv3d(in_channels = landsat_inc, out_channels = 10, kernel_size =kernel_size, stride=1, padding = 0) 
        self.pool = nn.MaxPool3d(kernel_size = kernel_size, stride =1, padding =0) 
        self.conv2 = nn.Conv3d(in_channels = 10, out_channels = 15, kernel_size = kernel_size, stride=1, padding = 0) 
        self.conv3 = nn.Conv3d(in_channels = 15, out_channels = 20, kernel_size = kernel_size, stride=1, padding = 0) 
        self.dropout = nn.Dropout(ec_dp)
        self.bn1 = nn.BatchNorm3d(10)
        self.bn2 = nn.BatchNorm3d(15)
        self.bn3 = nn.BatchNorm3d(20)        
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)   
        x = self.bn2(x)

        
        x = F.relu(self.conv3(x)) 
        x = self.pool(x)
        x = self.bn3(x)    

        x = self.dropout(x)        

        
        return x
  

 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(20, 20, kernel_size=(3,11), padding=(1,0), bias=False)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=(3,3), padding='same', bias=False)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=(3,11), padding=(1,0), bias=False)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=(3,3), padding='same', bias=False)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=(3,5), padding=(1,0), bias=False)
        self.conv6 = nn.Conv2d(20, 30, kernel_size=(1,3), padding=(0,0), bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=3, padding=kernel_size//2, bias=False)
           
            
            
            
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(30)       

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x1 = self.bn1(x1)
        x2 = F.elu(self.conv2(x1))
        x2 = self.bn1(x2)  
        res1 = torch.cat((x1,x2),3)
        
        x3 = F.elu(self.conv3(res1))
        x3 = self.bn1(x3)
        x4 = F.elu(self.conv4(x3))
        x4 = self.bn1(x4)       
        res2 = torch.cat((x3,x4),3)
        
        
        x5 = F.elu(self.conv5(res2))
        x5 = self.bn1(x5)       
        x6 = F.elu(self.conv6(x5))
        x6 = self.bn2(x6)
   
        avg_out = torch.mean(x6, dim=1, keepdim=True)
        max_out, _ = torch.max(x6, dim=1, keepdim=True)
        x7 = torch.cat([avg_out, max_out], dim=1)

        x8 = self.conv7(x7)   
        x9 = self.sigmoid(x8) 

        return x9

 
class spattn(nn.Module):

	def __init__(self):
		super(spattn, self).__init__()       
		self.sa = SpatialAttention() 

	def forward(self,x):
		nsamples, nc,nt, nx, ny = x.shape        
		spavect = torch.zeros([nt,nsamples, 1,nx, ny], dtype=torch.float)
		for i in range(nt):
		    spa_inp = x[:,:,i,:,:]
		    spa_att = self.sa(spa_inp)      
		    spavect[i] = spa_att
		spavect = spavect.permute(1,2,0,3,4)
		spavect = spavect.to(device)        
		return spavect

 
class ffd(nn.Module):


    def __init__(self, landsat_inc,lninp,sl,hidden_dim):
        super(ffd, self).__init__()      
        self.fc1 = nn.Linear(lninp, 2000) 
        self.fc2 = nn.Linear(2000, hidden_dim) 
        self.layernorm_1 = nn.LayerNorm(2000, elementwise_affine = False)
        self.layernorm_2 = nn.LayerNorm(hidden_dim,  elementwise_affine = False)
        
    def forward(self,x):
        
        nsamples, nc,nt, nx, ny = x.shape
        out = x.reshape((nsamples,nc*nt*nx*ny))
        

        ll_1 = self.fc1(out)
        sigmoid = nn.ReLU()
        out = sigmoid(ll_1)
        out = self.layernorm_1(out) 
        ll_2 = self.fc2(out)
        relu = nn.ReLU()
        out = relu(ll_2)
        out = self.layernorm_2(out)
        return out

 
def softmax_calc(x):
    max = torch.max(x)
    sub = torch.sub(x,max)
    e_x = torch.exp(sub)
    return e_x / torch.sum(x,axis=0)

 
def highest_att(y,vectors,grd_idxlist,num_patches,pns_idx,pns_scr,flg):
    topvectors = torch.zeros(vectors.shape[0],num_patches,vectors.shape[2])
    best_grididx  = []
    notusedscr = []
    norusedidx = []
    for j in range(0,vectors.shape[0]):
        ylst = y[j].tolist()
        if flg == 0:
            candidatescr = pns_scr+ylst
            candidateidx = pns_idx+grd_idxlist[j]
        else:    
            candidatescr = pns_scr[j]+ylst
            candidateidx = pns_idx[j]+grd_idxlist[j]

        num_patches = len(candidateidx)//2
        candidatescr = torch.tensor(candidatescr)
        indices = torch.topk(candidatescr,num_patches).indices
        totalcnd = list(range(0,len(candidateidx)))
        ntsidx = [item for item in totalcnd if item not in indices] 

        top_grididx = []
        nsscors = []
        ns_grdidx = []
        topvec = torch.zeros(num_patches,vectors.shape[2])
        for (i,idx) in enumerate(indices):

            grd  = candidateidx[idx]
            top_grididx.append(grd)      

        for (i,idx) in enumerate(ntsidx):
            nsgrd  = candidateidx[idx]
            nsc = candidatescr[idx]
            ns_grdidx.append(nsgrd)        
            nsscors.append(nsc)
        best_grididx.append(top_grididx)
        notusedscr.append(ns_grdidx)
        norusedidx.append(nsscors)
 
    return best_grididx, notusedscr, norusedidx

 
def next_grids(top_grididx, gridlist, image, ps, traversed_idx):
    
    total_grids = []
    for j in range(0,image.shape[0]):
        next_grids = []
        grdlist=  gridlist[j]

        for k in top_grididx[j]:
            grid_set = []
            selectedgrdlst = grdlist[k]

            row, col = selectedgrdlst
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < image[j].shape[2]//ps and 0 <= new_col < image[j].shape[3]//ps:
                        grid_set.append((new_row, new_col))
            next_grids.extend(grid_set)
            next_grids = next_grids[:24]
        total = list(range(0,len(grdlist)))
        untraversed_grdidx = [x for x in total if x not in traversed_idx[j]]
        indices = [grid[0] * (image[j].shape[3]//ps) + grid[1] for grid in next_grids]
        unique_indices = list(set(indices))
        while len(unique_indices )<30:
            cnt = 30-len(unique_indices)
            untrav_index = np.random.choice(len(untraversed_grdidx), size=cnt, replace=False)
            untrv = grdlist[untrav_index[0]]
            next_grids.append(untrv)
            indices = [grid[0] * (image[j].shape[3]//ps) + grid[1] for grid in next_grids]
            unique_indices = list(set(indices))
        if len(unique_indices )>30:
            unique_indices_fin =unique_indices[0:30]
        elif len(unique_indices )==30:
            unique_indices_fin = unique_indices    
        total_grids.append(unique_indices_fin)
    return total_grids

 
def attention_calc(vectors,w_q,w_k,w_v,n):

    Q = torch.multiply(vectors,torch.transpose(w_q,0,1))
    K = torch.multiply(vectors,torch.transpose(w_k,0,1))
    V = torch.multiply(vectors,torch.transpose(w_k,0,1))   
    attn_scores = Q @ K.T      
    attn_scores_softmax = softmax_calc(attn_scores)
    attn_scores_softmax = torch.tensor(attn_scores_softmax)
    outputs = attn_scores_softmax.sum(dim=0) 
    res = softmax_calc(outputs)
    res_reshape = torch.unsqueeze(res,1)
    weighted_values = res_reshape * V 

    return res, weighted_values

 
def initializeGrid(batch,X_range,Y_range):
    grid = []
    for k in range (0,batch):
        temp = []
        for i in range (0,X_range):
            for j in range (0,Y_range):
                temp.append([i,j])
        grid.append(temp)
    return grid

 
def getRandGrid(gridlist,no_patch):
    total_indices = len(gridlist[0])
    num_sets = len(gridlist)  
    n = no_patch  
    sets_of_indices = []  
    for _ in range(num_sets):
        selected_indices = random.sample(range(total_indices), n)
        sets_of_indices.append(selected_indices)
        
    return sets_of_indices

 
def getPatch(item,index,ps):
    patch = torch.zeros([item.shape[0],index.shape[1],item.shape[1],item.shape[2],ps,ps], dtype=torch.float)
    for k in range(0,item.shape[0]):
        ptchs = torch.zeros([index.shape[1],item.shape[1],item.shape[2],ps,ps], dtype=torch.float)
        for q in range(0,index.shape[1]):
            idnc = index[k,q]
            i = idnc[0]
            j = idnc[1]
            x_st = i*ps
            x_end = (i+1)*ps
            y_st = j*ps
            y_end = (j+1)*ps
            imx = item[k,:,:,x_st:x_end,:]
            imp = imx[:,:,:,y_st:y_end]
            imp = torch.tensor(imp)
            ptchs[q,:,:,:,:] = imp
        patch[k,:,:,:,:] = ptchs
    
    return patch

 
class mds_encoder(nn.Module):

    def __init__(self, landsat_inc,hidden_dim,sl,ec_dp):
        super(mds_encoder, self).__init__()
        
        self.cnn3d = cnn3d(landsat_inc,sl,ec_dp,kernel_size=(5,23,39))
        self.spttn = spattn('mod')
        self.ffd_mds = ffd(landsat_inc,(20*11*18*22),sl,hidden_dim)

        
    def forward(self,X):
        cnn_mds = self.cnn3d(X)
        spt_mds = self.spttn(cnn_mds)
        mds = torch.mul(cnn_mds,spt_mds)
        ffdmds_out = self.ffd_mds(mds)
        
        
        return ffdmds_out

 
class lnt_encoder(nn.Module):
    def __init__(self, landsat_inc,ps,hidden_dim,sl,loop_cnt,ec_dp):
        super(lnt_encoder, self).__init__()    
        
        
        self.cnn3d_lnt = cnn3d(landsat_inc,slm,dp,kernel_size=(3,9,9)) 
        self.embedll = nn.Linear(hidden_dim, hidden_dim)
        self.spttn = spattn('lnt')
        self.ffd_mds = ffd(landsat_inc,(20*(ps-48)*(ps-48)*(sl-12)),slm,hidden_dim)
        self.ps = ps
        self.loop_cnt = loop_cnt
        self.patch_cnt = 20    
        self.W_Q = torch.nn.Parameter(torch.zeros(hidden_dim, 1).to(device))
        torch.nn.init.xavier_normal_(self.W_Q)
        self.W_K = torch.nn.Parameter(torch.zeros(hidden_dim, 1).to(device))
        torch.nn.init.xavier_normal_(self.W_K)
        self.W_V = torch.nn.Parameter(torch.zeros(hidden_dim, 1).to(device))
        torch.nn.init.xavier_normal_(self.W_V)
        self.embedding = CrossModalSimilarityHashing(hidden_dim,4)#nn.Linear(200,5)#nn.Embedding(5, 2)   
        self.embedll = nn.Linear(680, hidden_dim).to(device)        
    def forward(self,X):
        ps = self.ps  
        X_range = X.shape[3]//ps        
        Y_range = X.shape[4]//ps                 
        loop_cnt  = self.loop_cnt
        
        gridlist = initializeGrid(X.shape[0],X_range,Y_range)
        gridtensor = torch.tensor(gridlist)         
        initial_gridlist = getRandGrid(gridlist,self.patch_cnt)        
      
        initial_grid = torch.tensor(initial_gridlist,dtype=int) 
        batchembeding = torch.zeros([X.shape[0],hidden_dim], dtype=torch.float)              
        traversed_grididx = []
        embedinglist = []        
        no_grids  = len(initial_gridlist[0])    
        trvrsedpxl  = 0
        nsgridlist = []    
        nsgridscores = []        
        while trvrsedpxl < loop_cnt:
            reshaped_initial_grid = initial_grid.unsqueeze(-1).expand(-1, -1, 2)
            selected_coordinates = torch.gather(gridtensor, 1, reshaped_initial_grid)
            patches = getPatch(X,selected_coordinates,ps)
            patches = patches.to(device)
            bs,pn,ch,ts,nx,ny = patches.shape
            totalpatch = patches.reshape(bs*pn,ch,ts,nx,ny)
            
            cnn_mds = self.cnn3d_lnt(totalpatch)
            spt_mds = self.spttn(cnn_mds)
            mds = torch.mul(cnn_mds,spt_mds)
            fdmds_out = self.ffd_mds(mds)  
            attention_val, enchced_ffd = attention_calc(fdmds_out,self.W_Q,self.W_K,self.W_V,no_grids)
            attention_val =   attention_val.reshape(X.shape[0],no_grids)      
            enhced_outreshape = enchced_ffd.reshape(X.shape[0],no_grids,fdmds_out.shape[1])
            topgridid, nsidx, nsscores = highest_att(attention_val,enhced_outreshape,initial_gridlist,no_grids,nsgridlist,nsgridscores,trvrsedpxl)
            nsgridlist=  nsidx
            nsgridscores= nsscores
            
            if len(traversed_grididx)==0:
                traversed_grididx += initial_gridlist

            else:
                for i in range(len(traversed_grididx)):
                    traversed_grididx[i] += initial_gridlist[i]
            trvrsedpxl  = trvrsedpxl+len(initial_gridlist[0])                    
            next_grdlistidx = next_grids(topgridid, gridlist, X,ps,traversed_grididx)
            initial_gridlist = next_grdlistidx
            no_grids = len(next_grdlistidx[0])            
            initial_grid = torch.tensor(initial_gridlist,dtype=int)
            attention_val = torch.unsqueeze(attention_val, 2)             
            embd_inp = enhced_outreshape
            embd_inp = (embd_inp).to(device)
            embedded_output = self.embedding(embd_inp)    
            ns,nx,ny = embedded_output.shape        
            embedded_outputresh = embedded_output.reshape(ns,nx*ny)
            embedinglist.append(embedded_outputresh)
            

        concat_embd = torch.cat(embedinglist, dim=1) 
        concat_embd = concat_embd.to(device)
        batchembeding = self.embedll(concat_embd)
              
        return batchembeding        
        
        

 
class CrossModalSimilarityHashing(nn.Module):
    def __init__(self, input_dim, hash_bits):
        super(CrossModalSimilarityHashing, self).__init__()
        self.fc = nn.Linear(input_dim, hash_bits)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        return x

 
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

 
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.1,
        talking_heads = False,
        prenorm = False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)


        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))


        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale


        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias


        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)


        attn = stable_softmax(sim, dim = -1)
        context_attn = stable_softmax(sim, dim = -2)


        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)


        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)


        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out

 
class patchguide(nn.Module):
    def __init__(self, modis, lnt):
        super().__init__()
        self.fusion_function = 'softmax'

    def forward(
        self,
        mds,
        lnt,
    ):
        lnt_score = torch.matmul(mds, lnt.transpose(-1, -2))       

        if self.fusion_function == 'softmax':
            lnt_prob = nn.Softmax(dim=-1)(lnt_score)
    
            lntout = torch.matmul(lnt_prob, lnt)
        
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
            
            
        return lntout

 
class fusion_module(nn.Module):

    def __init__(self, landsat_inc,psl,hidden_dim,slm,sll,dp,num_heads,loop_cnt,ec_dp):
        super(fusion_module, self).__init__()

        self.mds_enc = mds_encoder(landsat_inc,hidden_dim,slm,ec_dp)
        self.lnt_enc = lnt_encoder(landsat_inc,psl,hidden_dim,sll,loop_cnt,ec_dp)  
        self.joint_cross_attn = BidirectionalCrossAttention(dim=hidden_dim)
        self.linear_layer = nn.Linear((hidden_dim*3),hidden_dim,bias = True)
        self.lntmodguide   = patchguide(sll*hidden_dim,sll*hidden_dim) 

    def forward(self, X1, X2):
         
        mdout = self.mds_enc(X1)
        lsout = self.lnt_enc(X2)
        fusn_nois_lnt = self.lntmodguide(mdout,lsout)        
        mdout = mdout.unsqueeze(0)
        lsout = lsout.unsqueeze(0)        
        
        attn, crossatn_out_m = self.joint_cross_attn(mdout, lsout) 
        attn, crossatn_out_l = self.joint_cross_attn(lsout, mdout)                     
        crossatn_out_m = crossatn_out_m.squeeze(0)
        crossatn_out_l = crossatn_out_l.squeeze(0)
        mdout = mdout.squeeze(0)
        lsout = lsout.squeeze(0)
        modskip = F.relu(mdout+crossatn_out_m)
        lntskip = F.relu(lsout+crossatn_out_l)

        
        fusion_noise_out  = torch.cat((modskip,lntskip,fusn_nois_lnt), dim = 1)
        ll = self.linear_layer(fusion_noise_out)        
        sigmoid = nn.GELU()
        combined_out = sigmoid(ll)        
              
              
        return combined_out

 
class prediction_model(nn.Module):

	def __init__(self, landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,out_dim,loop_cnt,ec_dp):
		super(prediction_model, self).__init__()
		self.fusion = fusion_module(landsat_inc,psl,hidden_dim, slm,sll,dp,num_heads,loop_cnt,ec_dp)
		self.ll1_layer = nn.Linear(hidden_dim, hidden_dim//2) 
		self.ll2_layer = nn.Linear(hidden_dim//2, hidden_dim//4)         
		self.output_layer = nn.Linear(hidden_dim//4, out_dim) 

	def forward(self,X1,X2):
        
		combined_out = self.fusion(X1,X2)          
		ll1out = self.ll1_layer(combined_out)
		ll2out = self.ll2_layer(ll1out)      
		out = self.output_layer(ll2out)        
		return out

 
class contrastive_model(nn.Module):

    def __init__(self, landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,out_dim,loop_cnt):
        super(contrastive_model, self).__init__()

        self.fusion_anc = prediction_model(landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,out_dim,loop_cnt,0.3)
        self.fusion_pos = prediction_model(landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,out_dim,loop_cnt,0.7)

    def forward(self, X1, X2,X3,X4,X5,X6):

        
        anc_out = self.fusion_anc(X1,X2)
        pos_out = self.fusion_pos(X3,X4)
        neg_out = self.fusion_anc(X5,X6)
        
        return anc_out,pos_out,neg_out

 
path_to_save = '.../path to save models'
path_to_save

 
def make_pairs(st_m,cnt_m,m_yr,indix_stm,indix_cntm,year_m):

    pos_idxs = []
    neg_idxs = []
    total_idx = list(range(0,len(indix_cntm)))
    
    for i in range(0,len(indix_cntm)):
        posidxs = torch.where((indix_stm[:] == st_m) &(indix_cntm[:] == cnt_m) &(year_m[:] == m_yr))[0]
        posidxs = posidxs.cpu().numpy()                
        pos_id  =  np.random.choice(posidxs)
        negIdxs = torch.where((indix_stm[:] != st_m) |(indix_cntm[:] != cnt_m)|(year_m[:] != m_yr))[0]
        negIdxs = negIdxs.cpu().numpy()
        if len(negIdxs)==0:
            negIdxs = posidxs
        neg_id  =  np.random.choice(negIdxs)
        
       
    return [pos_id, neg_id]

 
def evaluate(model,data_loader, dataset):
    model.eval()
    
    eval_loss1 = float(0) 
    eval_loss2 = float(0)  
    
    correct = 0
    avg_loss = []
    output_list = []
    target_list = []
    count = 0
    model = model.to(device)
    with torch.no_grad():
        for data_m, data_l, st_m,cnt_m,yr_m, target in data_loader:
            data_m, data_l, st_m,cnt_m,yr_m, target = (data_m.float()).to(device),(data_l.float()).to(device),(st_m.float()).to(device),(cnt_m.float()).to(device),(yr_m.float()).to(device),target.float().to(device)
            output, output_pos, output_neg = model(data_m, data_l,data_m, data_l,data_m, data_l)
            target = target.squeeze(-1) 
            output = output.squeeze(-1) 
            eval_lossb1 = criterion_print(output, target).mean(axis=0)*len(target)
            eval_lossb1 = torch.tensor(eval_lossb1)
            eval_loss1 += eval_lossb1        
            count += len(target)
            output_list += output.tolist()
            target_list += target.tolist()
            
    eval_loss1 /= count
    eval_loss1 = torch.sqrt(torch.tensor(eval_loss1))    
    eval_loss1 = eval_loss1.cpu().numpy()
   
    print('{} set: loss1: {}\n'.format(dataset,  eval_loss2))
    if dataset == 'Validation':
      return eval_loss1, output_list, target_list
    if dataset == 'Test':
      return eval_loss1, output_list, target_list

 
def train(model, optimizer,epcs):
	cl_loss1= []
	cl_loss2= [] 
	cls_loss= []     
	mse_loss = []
    
	history_train = []
	history_val1 = []
	history_val2 = []
   
	output_train = []
	target_train = []

	for epoch in range(1, epcs+1):

		model.train()
      
		for batch_idx, (data_m, data_l, st_m,cnt_m,yr_m, target) in enumerate(train_dataloader):

			data_m, data_l, st_m,cnt_m,yr_m, target = (data_m.float()).to(device),(data_l.float()).to(device),(st_m.float()).to(device),(cnt_m.float()).to(device),(yr_m.float()).to(device),target.float().to(device)
			optimizer.zero_grad(set_to_none = True)     
			positive_pairs = []
			negative_pairs = []                        
			for i in range(0,len(target)):
			    positive_idx, negative_idx = make_pairs(st_m[i],cnt_m[i], yr_m[i],st_m,cnt_m,yr_m)            
			    positive_pairs.append(positive_idx)
			    negative_pairs.append(negative_idx)                

			data_m_pos = data_m[positive_pairs]
			data_l_pos = data_l[positive_pairs]
			lnt_pos = data_l_pos
			mds_pos = data_m_pos   

			anc_output, positive_out, negative_out= model(data_m, data_l,mds_pos,lnt_pos,data_m[negative_pairs], data_l[negative_pairs])  
			target = target.repeat(anc_output.shape[1], 1).T
			output = anc_output.squeeze(-1) 
            
			loss_1 = torch.sqrt(criterion_pred(output,target))
			loss_2 = criterion_triplet(anc_output, positive_out,target)
			loss_3 = criterion_triplet(anc_output, negative_out,target)            
			loss = loss_1 +loss_2 +loss_3

			loss.backward()
			optimizer.step()

		if epoch >= 0:
		  torch.save(model, os.path.join(path_to_save, "model_soy_%d.pt"%(epoch)))   
		print('Train Epoch: {} Training Loss: {}'.format(epoch,loss.item()))

		result_val1,train_preds, labels = evaluate(model,val_dataloader,'Validation')

		mse_loss.append(loss_1.item()) 
		history_train.append(loss.item())
		history_val1.append(result_val1)       
		output_train.append(train_preds)
		target_train.append(labels) 

        
	return history_train, history_val1,output_train, target_train, mse_loss # ,cls_loss,    
    

 
def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
)
    if type(module) == nn.Conv3d:    
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.001)

 
landsat_inc = 5

slm = len(timesteps_m)
sll = len(timesteps_l)
sl  = len(timesteps_l)
hidden_dim = 100
epochs = 51   # 75
n_layers = 1
output_dim = 1
psl = 64

num_heads = 4
d_model = hidden_dim
dp = 0.25
loop_cnt  = ((2240//64)*(2880//64))//4
loop_cnt

 
class MarginBasedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginBasedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(torch.clamp(target * euclidean_distance, min=0.0, max=self.margin))
        return loss

 


model_pred = contrastive_model(landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,output_dim,loop_cnt).to(device)
model_pred.apply(init_weights)


criterion_triplet = MarginBasedContrastiveLoss(margin=0.5)
criterion_pred = nn.MSELoss().to(device)
criterion_print = nn.MSELoss(reduce = False, reduction = 'None').to(device)

optimizer = optim.Adam(model_pred.parameters(), lr=0.000005)

 
history_t, history_v1,  output_train, target_train,  ms_l = train(model_pred, optimizer, epochs) # cl_l2,cl_l1, cl_l2,

 
losses1 = []
epochs = 5

mdls = []
for m in range(3,epochs+1):
    model_trained=torch.load(os.path.join(path_to_save, "model_soy_%d.pt"%(m)))
    loss1, preds, labels = evaluate(model_trained, test_dataloader, 'Test')
    mdls.append(m)
print( min(losses1), min(losses1) )

 
best = mdls[losses1.index(min(losses1))]
best 

 
model_trained=torch.load(os.path.join(path_to_save, "model_soy_%d.pt"%(best)))
loss1, preds, labels = evaluate(model_trained, test_dataloader, 'Test')
loss_met_0 = loss1

print('loss_met_0', loss_met_0)

 
fin_label = np.array(labels)
fin_label = fin_label
fin_preds = np.array(preds)
fin_preds = fin_preds

 
df_test = pd.DataFrame(tststates, columns=['State'])
df_test['county'] = tstcounties
df_test['Actual'] = fin_label
df_test['Predicted'] = fin_preds
df_test[0:12]

 
# import sklearn
from sklearn import metrics
from sklearn.metrics import r2_score 
rsquare_vis = r2_score(fin_label,preds)
mae_vis = metrics.mean_absolute_error(labels,preds)

 
rmse_df = pd.DataFrame({'loss_1': loss1, 'r_square':rsquare_vis,'mae':mae_vis}, index = ['lr_000001_'+crop+'_'+mdl+'_'+str(tst_year)])
rmse_df

 
rmse_df_ep= pd.DataFrame({'loss': loss1}, index = ['rmse2_0_'+crop+'_'+mdl+'_'+str(tst_year)])


rmse_df_ep

 
df_test.to_csv('.../Predictions.csv') #path to save predictions file
rmse_df.to_csv('.../error.csv') #path to save error file

print('csv created for '+crop+'_'+mdl)

 


 


