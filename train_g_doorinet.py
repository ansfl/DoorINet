###########################   IMPORTS   #############################

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from utils.data_processing import *
import datetime
from sklearn.model_selection import train_test_split
from models.models_definitions import *
from models.models_training import *


###########################   CONSTANTS   ############################

if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device("cuda")

DATA_FOLDER = './dataset/'
MODELS_FOLDER = './models/'
BATCH_SIZE = 64
T = 20
step = 20


###########################    MAIN    ##############################
def main(SEED=None):
    print('G-DoorINet training script')
    if SEED is not None:
        SEED = round(float(SEED))
        print('using seed', SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load train data
    print('loading training data...')
    # print(datetime.datetime.now())
    order = ['gyr_x','gyr_y','gyr_z','acc_x','acc_y','acc_z']

    data_file = os.path.join(DATA_FOLDER, 'train_dataset.csv')
    data = pd.read_csv(data_file)
    data_x = data[order].copy()
    data_y = np.array(data['heading_angle'])

    X = np.zeros((data_x.shape[0] // step, 6, T))
    Y = np.zeros((data_y.shape[0] // step, 1))

    n = 0
    ind_order = []
    for o in order:
        ind_order.append(data_x.columns.str.contains(o).argmax())
    x_np = np.array(data_x)
    for t in range(0, data_x.shape[0] - T, step):
    #     0:3 - gyr
    #     3:6 - acc
        X[n, :, :] = x_np[t:t + T, ind_order].T.copy()
        Y[n] = data_y[t+T] - data_y[t]
        n = n + 1

    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))

    x_train, x_val, y_train, y_val = train_test_split(X, Y, shuffle=True, test_size=0.2, random_state=SEED)

    x_train = torch.utils.data.DataLoader(dataset=x_train, batch_size=BATCH_SIZE)
    x_val = torch.utils.data.DataLoader(dataset=x_val, batch_size=BATCH_SIZE)
    y_train = torch.utils.data.DataLoader(dataset=y_train, batch_size=BATCH_SIZE)
    y_val = torch.utils.data.DataLoader(dataset=y_val, batch_size=BATCH_SIZE)

    print('training the model...')
    model_name = 'G-DoorINet'
    g_doorinet = G_DoorINet(drop_out=0.1, model_name=model_name).to(DEVICE)
    # initialize weights
    for name, param in g_doorinet.named_parameters():
        if param.dim()>=2:
            nn.init.xavier_uniform_(param)
    params = sum(p.numel() for p in g_doorinet.parameters() if p.requires_grad)*1e-3
    paramstr = '{mod_name} has {parameters:.2f}k parameters\n'
    print(paramstr.format(mod_name=model_name, parameters=params))
    optimizer = torch.optim.AdamW(params=g_doorinet.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, threshold_mode='rel',cooldown=1,threshold=0.01)
    loss_fn = nn.HuberLoss()

    history = train(x_train, x_val, y_train, y_val, model=g_doorinet, criterion=loss_fn, epochs=150, sched=scheduler, opt=optimizer, 
                    DEVICE=DEVICE, start_epoch=0)

    print('saving the model...')
    model_fname = os.path.join(MODELS_FOLDER, 'G_DoorINet_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'_seed_'+str(SEED)+'.pt')

    torch.save(g_doorinet.state_dict(), model_fname)
    print('G-DoorINet model saved as', model_fname)
    print('done!')

if __name__=="__main__":
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        main()