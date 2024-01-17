import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from models.models_definitions import *
from utils.data_processing import *



XDOT_SAMPLE_RATE = 120

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_g_doorinet(filename, DEVICE, name='G-DoorINet'):
    g_doorinet = G_DoorINet(drop_out=0.1).to(DEVICE)
    g_doorinet.model_name = name
    g_doorinet.load_state_dict(torch.load(filename))
    g_doorinet.eval()
    return g_doorinet


def load_ag_doorinet(filename, DEVICE, name='AG-DoorINet'):
    ag_doorinet = AG_DoorINet(drop_out=0.1).to(DEVICE)
    ag_doorinet.model_name = name
    ag_doorinet.load_state_dict(torch.load(filename))
    ag_doorinet.eval()
    return ag_doorinet


def load_gm_doorinet(filename, DEVICE, name='GM-DoorINet'):
    gm_doorinet = GM_DoorINet(drop_out=0.1).to(DEVICE)
    gm_doorinet.model_name = name
    gm_doorinet.load_state_dict(torch.load(filename))
    gm_doorinet.eval()
    return gm_doorinet



def test_epoch(model, x_test, y_test, DEVICE, criterion,  subtract_mean=False):
   
    '''
    subtract_mean: mean of all the test dataset is subtracted from the output (strange hack)

    '''
    model.eval()
    total_out = np.empty(shape=[0,0])
    total_sec = np.array([])
    running_loss = 0.0
    processed_size = 0
    t_errors = np.empty([])
    # ground_truth = np.empty([])
    mse = np.empty([])
    for b_x, b_y in zip(x_test, y_test):
        b_acc = b_x[:, 3:6].to(DEVICE)
        b_gyro = b_x[:, 0:3].to(DEVICE)
        b_mag = b_x[:,6:9].to(DEVICE)
        b_y = b_y.to(DEVICE)
        # ground_truth = np.append(ground_truth, b_y.cpu())
        with torch.set_grad_enabled(False):
            if model.magnetic:
                b_out = model(b_mag,b_gyro)
            if not(model.gyro_only):
                b_out = model(b_acc, b_gyro)
            else:
               b_out = model(b_gyro)
            loss = criterion(b_out, b_y)
        total_out = np.append(total_out, b_out.cpu())
        t_errors = np.append(t_errors,(-b_out.cpu()+b_y.cpu()).numpy()) 
        mse= np.append(mse,((b_out.cpu()-b_y.cpu())**2).numpy())
        total_sec = np.append(total_sec,b_x[:,-1,-1].numpy())
        running_loss += loss.item() * b_gyro.size(0)
        processed_size += b_y.size(0)
        del b_acc
        del b_gyro
        del b_mag
        del b_y
        del b_out
    test_loss = running_loss / processed_size
    angles = [0]
    if subtract_mean:
        total_out -= total_out.mean()
        # print('Output mean was subtracted!')
    for i in range(1, len(total_out)+1):
        angles = np.append(angles, angles[-1]+total_out[i-1]) 
    return test_loss, total_out, total_sec, angles[1:], t_errors[1:]


def test_nn_model_long(models, x_test, y_test, DEVICE, imu_data=None, criterion=nn.HuberLoss(), ext_data=None, show_madgwick=True, figsize=(20,6), show_integration=True, subtract_mean=False,
                       gt_resampled=None, return_data=False, show_range=None, cut_last_points=None, legend_loc='best', ncol=0, data_name='', imu_number=0):
    '''
    tests NN model on long test data with the index N with criterion
    cut_last_points - number of points to cut from the right side
    '''
    max_angle = 0
    nn_mses = []
    nn_rmses = []
    nn_diffs = []
    nn_angles = []
    nn_test_losses = []
    nn_errors = []
    nn_sec = []   # seconds to plot results
    nn_outputs = []
    nn_rmses = []
    last_nn_diffs = []
    nn_maxerrors = []
    test_losses = []
    sec = None
    gt_resampled = None 
    gt_plotted = False
    
    # resample ground_truth
#     sec_xdot = xdot_test_X[n]['seconds'].copy()
    print('Model name\t\tMSE\t\tRMSE\t\tLPD\t\tMAD\t\tloss')
    fig,axes = plt.subplots(1,1,figsize=figsize,sharex=True,tight_layout=True)

    # axes[1].set_title('Difference from ground truth', size='x-large')

    for i,model in enumerate(models):
        test_loss, total_out, sec, angles, errors = test_epoch(model, x_test, y_test, DEVICE, criterion=criterion, subtract_mean=subtract_mean)          
        if cut_last_points is not None:
#             print('sec before cutting:', sec.shape)
            sec  = sec[:-cut_last_points]
            angles = angles[:-cut_last_points]
            errors = errors[:-cut_last_points]
#             print('sec after cutting:', sec.shape)
        
        test_losses.append(test_loss)
        nn_outputs.append(total_out)
        nn_sec.append(sec)
        nn_test_losses.append(test_loss)
        nn_errors.append(errors)
         
        gt_resampled = find_closest(sec, imu_data['seconds'], imu_data['heading_angle'])
        # gt_resampled = angles

        if not gt_plotted:
            axes.scatter(sec, gt_resampled, s=0.5, label='Ground truth')
            gt_plotted = True
            ncol += 1

#         print('angles shape', len(angles))
#         print('sec shape', len(sec))
#       calculate errors like other methods (madgwick and direct integration)
        nn_diff = gt_resampled - angles
        nn_mse = np.mean(nn_diff**2)
        nn_rmse = np.sqrt(nn_mse)
#         nn_mse = np.mean(errors**2)
        nn_mses.append(nn_mse)
#         nn_rmse = np.sqrt(nn_mses[-1])
        nn_rmses.append(nn_rmse)

        nn_diffs.append(nn_diff)
        last_nn_diff = abs(nn_diff[-1])
        last_nn_diffs.append(last_nn_diff)
        nn_maxerror = np.max(np.abs(nn_diff))
        nn_maxerrors.append(nn_maxerror)
    
        print('{model_name}\t\t{mse:.4f}\t\t{rmse:.4f}\t\t{last:.4f}\t\t{ermax:.4f}\t\t{loss:.4f}'.format(model_name=model.model_name, 
                                                                mse=nn_mse,rmse=nn_rmse,last=last_nn_diff,ermax=nn_maxerror,loss=test_loss))
        axes.scatter(sec, angles, s=0.5, label=model.model_name)
        max_angle = np.max([max_angle,np.max(angles), np.max(gt_resampled)])
        ncol += 1
        
    if ext_data is not None:
        for i,data in enumerate(ext_data):
#             resample?

            if cut_last_points is not None:
                e_data = data[1][:,:-cut_last_points]
            else:
                e_data = data[1][:,:]
            data_name = data[0]
            if sec is not None:
                data_resampled = find_closest(sec, e_data[0,:], e_data[1,:])
                data_resampled[0] = 0
            else:
                sec = e_data[:, 0]
                data_resampled = e_data[:,1]     
                
#             calc mse + rmse + 
            if gt_resampled is None:
                gt_resampled = find_closest(sec, imu_data['seconds'], imu_data['heading_angle'])
            common_length = min(len(gt_resampled), len(data_resampled))
            d_diff = gt_resampled[:common_length] - data_resampled[:common_length]
            d_mse = np.mean(d_diff**2)
            d_rmse = np.sqrt(d_mse)
            d_aad = abs(d_diff[-1])
            d_mad = np.max(abs(d_diff))                
#             plot 
            
            print('{d_name}\t\t{mse:.4f}\t\t{rmse:.4f}\t\t{last:.4f}\t\t{ermax:.4f}'.format(d_name=data_name,mse=d_mse,rmse=d_rmse,last=d_aad,ermax=d_mad))
            lab = data_name
            axes.scatter(sec[:common_length], data_resampled[:common_length], s=0.5, label=lab)
            max_angle = np.max([max_angle,np.max(data_resampled)])
            ncol += 1
    
    if show_madgwick:
        madg_nomag_result = madgwick_eulers_no_mag(imu_data,XDOT_SAMPLE_RATE, gain=0.1, sec=5, acc_rej=10)[:,2]
        madg_res_corr = process_angles_for_plot(madg_nomag_result)
#         find indexes from 1 to 3 seconds (for my test sequence this is stationary conditions)
        indexes = np.array(imu_data[(imu_data['seconds']>1) & (imu_data['seconds']<3)].index) - imu_data.index[0]
#         subtract stationary heading to make "start from zero"
        mean_zero = np.mean(madg_res_corr[indexes])
        madg_res_corr -= mean_zero
        madg_res_corr[0:10] = 0 # this is for the first several points - it is usually very high or low
        madg_res_resampled = find_closest(sec, imu_data['seconds'], madg_res_corr)
        m_diff = gt_resampled - madg_res_resampled
        m_mse = np.mean(m_diff**2)
        m_rmse = np.sqrt(m_mse)
        last_m_diff = abs(m_diff[-1])
        m_maxerror = np.max(np.abs(m_diff))
        print('Madgwick NOMAG\t\t{mse:.4f}\t\t{rmse:.4f}\t\t{last:.4f}\t\t{ermax:.4f}'.format(mse=m_mse,rmse=m_rmse,last=last_m_diff,ermax=m_maxerror))
        axes.scatter(sec, madg_res_resampled, s=0.5, label='Madgwick')
        max_angle = np.max([max_angle, np.max(madg_res_resampled)])
        ncol += 1
    
    if show_integration:
        int_result = simple_integration(np.array(imu_data['gyr_y']), XDOT_SAMPLE_RATE)
        int_resampled = find_closest(sec,imu_data['seconds'], int_result)
        i_diff = gt_resampled - int_resampled
        i_mse = np.mean(i_diff**2)
        i_rmse = np.sqrt(i_mse)
        last_i_diff = abs(i_diff[-1])
        i_maxerror = np.max(np.abs(i_diff))
        print('Integration\t\t{mse:.4f}\t{rmse:.4f}\t\t{last:.4f}\t\t{ermax:.4f}'.format(mse=i_mse,rmse=i_rmse,last=last_i_diff,ermax=i_maxerror))
        axes.scatter(sec, int_resampled, s=0.5, label='Integration')
        max_angle = np.max([max_angle, np.max(int_resampled)])
        ncol += 1
    
    axes.grid(visible=True)
    if ncol > 8: ncol = 8
    lgnd = axes.legend(loc=legend_loc, fontsize=12, ncol=ncol)
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [30]
    axes.set_ylabel('Heading angle, [deg]',size='x-large')
    axes.set_xlabel('Time, [s]',size='x-large')
    if show_range is not None:
        left, right = show_range
        axes.set_xlim(left, right)
    s_title = 'Model output IMU #' + str(imu_number)
    plt.suptitle(s_title,y=0.95, size='x-large')
    plt.xticks(size='x-large')
    axes.yaxis.set_tick_params(labelsize=15)
    # axes.set_ylim([None, max_angle+10])
    plt.yticks(size='x-large')
    plt.draw()
    plt.show(block=False)
    if return_data:
        return sec, data_resampled