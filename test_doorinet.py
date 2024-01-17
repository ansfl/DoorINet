#############################    IMPORTS    #############################

import numpy as np
import pandas as pd
import glob, os
import torch
import torch.nn as nn
from models.models_definitions import *
from models.models_testing import *
from matplotlib import pyplot as plt


#############################    CONSTANTS    ############################

if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device("cuda")

DATA_FOLDER = './dataset/'
MODELS_FOLDER = './models/'
SEED = 0
BATCH_SIZE = 64
T = 20  
step = 20


#############################      MAIN      #############################


# get DoorINet models

def main():
    current_dir = os.getcwd()
    os.chdir(MODELS_FOLDER)
    print("load models:")
    models = []
    # g_count = 0
    # ag_count = 0
    for file in glob.glob("*DoorINet*.pt"):
        # print(file)
        if file.startswith('AG_DoorINet'):
            # ag_count += 1
            model_name = 'AG_DoorINet'
            # model_name = 'AG_DoorINet' + str(ag_count)
            model = load_ag_doorinet(file, DEVICE, name=model_name)
            print(model_name ,end='')

        elif file.startswith('G_DoorINet'):
            # g_count += 1
            model_name = 'G_DoorINet'
            # model_name = 'G_DoorINet' + str(g_count)
            model = load_g_doorinet(file, DEVICE,  name=model_name)
            print(model_name,end='')
        models.append(model)
        print(' model found in file', file)
    print('done loading models!')

    # load datasets
    os.chdir(current_dir)
    os.chdir(DATA_FOLDER)
    x_test = []
    y_test = []

    for file in glob.glob('*test*.csv'):
        print('\nloading test dataset from file', file)
        data = pd.read_csv(file)
        print('done')

        print('preparing the test dataset from file', file)
        xdot_sec = np.zeros(data.shape[0])
        xdot_sec[0] = 0
        sample_time = np.array(data['sampletimefine'])
        for i in range(1,data.shape[0]):
            if sample_time[i] == 0:
                xdot_sec[i] = xdot_sec[i-1] + 1/XDOT_SAMPLE_RATE
            elif abs(sample_time[i] - sample_time[i-1]) <1e6:
                xdot_sec[i] = xdot_sec[i-1] + (sample_time[i] - sample_time[i-1])/1e6
            else:
                xdot_sec[i] = xdot_sec[i-1] + (sample_time[i-1] - sample_time[i-2])/1e6
        data['seconds']= xdot_sec

        order = ['gyr_x','gyr_y','gyr_z','acc_x','acc_y','acc_z','mag_x','mag_y','mag_z','seconds']
        data_x = data[order].copy()
        data_y = np.array(data['heading_angle'])
        X = np.zeros((data_x.shape[0] // step, 10, T))
        Y = np.zeros((data_y.shape[0] // step, 1))
        # xdot_test_X.append(temp_test_X)
    
        ind_order = []
        for o in order:
            ind_order.append(data_x.columns.str.contains(o).argmax())
        n = 0
        x_np = np.array(data_x)
        for t in range(0, data_x.shape[0] - T, step):   
    #     0:3 - gyr
    #     3:6 - acc
    #     6:9 - mag
    #     9 - sec
            X[n, :, :] = x_np[t:t + T, ind_order].T.copy()
            Y[n] = data_y[t+T] - data_y[t]
            n = n + 1
        
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32))  

        x_test.append(torch.utils.data.DataLoader(dataset=X, batch_size=BATCH_SIZE))
        y_test.append(torch.utils.data.DataLoader(dataset=Y, batch_size=BATCH_SIZE))
        print('done')
        
        print('testing DoorINet models on a test dataset from', file)
        imu_number = int(''.join(filter(str.isdigit, file)))

        
        # # loading DAE results
        # dae_out_file = glob.glob('dae_*'+str(imu_number)+'*')[0]
        # dae = pd.read_csv(dae_out_file, index_col=0)
        # dae['data'] = process_angles_for_plot(dae['data'] * 180 / np.pi)
        # dae_data = 'DAE', dae.to_numpy().T

        # # loading Model_A results
        # #  Quaternion Model A
        # quat_res_file = glob.glob('quat_pred*'+str(imu_number)+'*')[0]
        # quat_res = np.loadtxt(quat_res_file,delimiter=',')
        # quat_sec_file = glob.glob('sec_quat*'+str(imu_number)+'*')[0]
        # quat_sec = np.loadtxt(quat_sec_file,delimiter=',')
        # modela_data = 'Quaternion ModelA', np.array([quat_sec, quat_res])

        # lgc_name = 'lgc_imu' + str(imu_number)
        # lgc_file_name = glob.glob(lgc_name+'.csv')[0]
        # lgc_sec_file_name = 'lgc_imu' + str(imu_number) + '_sec' + '.csv'
        # print('lgc_data', lgc_file_name)
        # print('lgc_sec', lgc_sec_file_name)
        # lgc_res_data = np.loadtxt(lgc_file_name, delimiter=',')
        # lgc_sec_data = np.loadtxt(lgc_sec_file_name, delimiter=',')
        # lgc_data = 'LGC-Net', np.array([lgc_sec_data, lgc_res_data])

        test_nn_model_long(models,x_test[-1], y_test[-1], DEVICE, data, ext_data=None, show_integration=True, show_madgwick=True, subtract_mean=True,
            data_name=file, imu_number=imu_number)

    plt.show()

    os.chdir(current_dir)
    print('finish')


######### MAIN EXECUTION #############
if __name__ == "__main__":
    main()
