import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dir_path, file_names, startyear, no_samples, ls_path, ls_list, timesteps_m, timesteps_l):
        self.img_dir = dir_path
        self.img_list = file_names
        self.no_samples = no_samples 
        self.startyear = startyear
        self.ls_path = ls_path
        self.ls_list = ls_list
        self.timesteps_m = timesteps_m
        self.timesteps_l = timesteps_l

    def __len__(self):
        return self.no_samples * len(self.img_list)

    def __getitem__(self, idx):
        file_num_md = idx // self.no_samples
        file_name = self.img_list[file_num_md]
        sample_num = idx % self.no_samples + self.startyear

        img_path = os.path.join(self.img_dir, file_name)
        files = np.load(img_path, allow_pickle=True)
        states = torch.tensor(files[files.files[0]])
        counties = torch.tensor(files[files.files[1]])
        years = torch.tensor(files[files.files[2]])
        red_img = torch.tensor(files[files.files[3]])
        blue_img = torch.tensor(files[files.files[4]])
        green_img = torch.tensor(files[files.files[5]])
        nir_img = torch.tensor(files[files.files[6]])
        swir_img = torch.tensor(files[files.files[7]])
        yld_img = torch.tensor(files[files.files[8]])

        img_state = states[0].numpy()
        img_county = counties[0].numpy()
        img_state = np.asarray(img_state, dtype='int')
        img_county = np.asarray(img_county, dtype='int')

        red_img = red_img[:, self.timesteps_m, :, :]
        blue_img = blue_img[:, self.timesteps_m, :, :]
        green_img = green_img[:, self.timesteps_m, :, :]
        nir_img = nir_img[:, self.timesteps_m, :, :]
        swir_img = swir_img[:, self.timesteps_m, :, :]

        states = states[sample_num]
        counties = counties[sample_num]
        red_img = red_img[sample_num]
        blue_img = blue_img[sample_num]
        green_img = green_img[sample_num]
        nir_img = nir_img[sample_num]
        swir_img = swir_img[sample_num]
        years = years[sample_num]

        states = torch.unsqueeze(states, 0)
        counties = torch.unsqueeze(counties, 0)
        years = torch.unsqueeze(years, 0)
        yld_img = torch.unsqueeze(yld_img[0], 0)

        md_img = torch.stack((red_img, blue_img, green_img, nir_img, swir_img), 0)
        md_img = md_img / 255
        md_img = np.array(md_img, dtype=np.float16)
        md_img = torch.tensor(md_img)

        file_num_ls = idx

        ls_file = self.ls_list[file_num_ls]

        ls_img_path = os.path.join(self.ls_path, ls_file)

        ls_f = np.load(ls_img_path, allow_pickle=True)

        state_l1 = torch.tensor(ls_f['output_state'])
        county_l1 = torch.tensor(ls_f['output_county'])
        year_l1 = torch.tensor(ls_f['output_year'])

        red_l = torch.tensor(ls_f['output_red'])
        blue_l = torch.tensor(ls_f['output_green'])
        green_l = torch.tensor(ls_f['output_blue'])
        nir_l = torch.tensor(ls_f['output_nir'])
        swir_l = torch.tensor(ls_f['output_swir'])
        yld_l = torch.tensor(ls_f['output_yld'])

        red_l = red_l[self.timesteps_l, :, :]
        blue_l = blue_l[self.timesteps_l, :, :]
        green_l = green_l[self.timesteps_l, :, :]
        nir_l = nir_l[self.timesteps_l, :, :]
        swir_l = swir_l[self.timesteps_l, :, :]

        state_l = torch.unsqueeze(state_l1[0], 0)
        county_l = torch.unsqueeze(county_l1[0], 0)
        year_l = torch.unsqueeze(year_l1[0], 0)
        yld_l = torch.unsqueeze(yld_l[0], 0)

        ls_img = torch.stack((red_l, blue_l, green_l, nir_l, swir_l), 0)
        ls_img = ls_img / 255
        ls_img = np.array(ls_img, dtype=np.float16)
        ls_img = torch.tensor(ls_img)

        return md_img, ls_img, states, counties, years, yld_l



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

def process_state(dir_list):
    states = []
    counties = []

    for file in dir_list:
        locations = file[:-4].split("_")
        state = int(locations[0])
        states.append(state)
        stat = unique(states)

    return stat

def process_county(dir_list, state):
    states = []
    counties = []

    for file in dir_list:
        locations = file[:-4].split("_")
        state_fl = int(locations[0])

        if state_fl != state:
            continue
        elif state_fl == state:
            county_fl = int(locations[1])
        counties.append(county_fl)
        county = unique(counties)
    return county

def unique(list1):
    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
