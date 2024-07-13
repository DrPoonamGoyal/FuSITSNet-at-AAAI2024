# main.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CropDataset_train, CropDataset
from model import FuSITSNet
from train import train, evaluate

# Assuming CUDA is available, otherwise set to 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    # Load your data here
    train_data = CropDataset_train(x_train, y_train, p_train, q_train, s_train, t_train)
    eval_data = CropDataset(x_eval, y_eval, p_eval, q_eval, s_eval, t_eval)

    # Initialize data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False)

    # Initialize model
    model = FuSITSNet(landsat_inc,psl,hidden_dim,sl,slm,sll,dp,num_heads,output_dim,loop_cnt)
    model.to(device)

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    train(model, optimizer, train_loader, eval_loader, epochs)

if __name__ == "__main__":
    main()
