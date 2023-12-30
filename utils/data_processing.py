import numpy as np
import pandas as pd
import imufusion


def get_noise_stats(df, winsize=50):
    '''
    functions calculates statistics of the noise of a signal that's supposed to represent static conditions
    Returns noise_stats:
    [[[mean_ax, mean_ay, mean_az],
     [mean_gx, mean_gy, mean_gz]],

     [[std_ax, std_ay, std_az],
     [std_gx, std_gy, std_gz]]]

     1st index: 0-mean, 1-std
     2nd index: 0-acc, 1-gyro
     3rd index: 0-x, 1-y, 2-z
    '''
    result = np.zeros(shape=(2,2,3))
    result[0][0][0] = df['acc_x'].iloc[:winsize].mean()
    result[0][0][1] = df['acc_y'].iloc[:winsize].mean()
    result[0][0][2] = df['acc_z'].iloc[:winsize].mean()
    result[0][1][0] = df['gyr_x'].iloc[:winsize].mean()
    result[0][1][1] = df['gyr_y'].iloc[:winsize].mean()
    result[0][1][2] = df['gyr_z'].iloc[:winsize].mean()
    result[1][0][0] = stdev(df['acc_x'].iloc[:winsize])
    result[1][0][1] = stdev(df['acc_y'].iloc[:winsize])
    result[1][0][2] = stdev(df['acc_z'].iloc[:winsize])
    result[1][1][0] = stdev(df['gyr_x'].iloc[:winsize])
    result[1][1][1] = stdev(df['gyr_y'].iloc[:winsize])
    result[1][1][2] = stdev(df['gyr_z'].iloc[:winsize])
    return result


def add_noise(df, winsize=50, noise_stats=None, num_noise=2000, reduce_noise=False, reduce_factor=2):
    '''
    noise_stats:
    [[[mean_ax, mean_ay, mean_az],
     [mean_gx, mean_gy, mean_gz]],

     [[std_ax, std_ay, std_az],
     [std_gx, std_gy, std_gz]]]

     1st index: 0-mean, 1-std
     2nd index: 0-acc, 1-gyro
     3rd index: 0-x, 1-y, 2-z
    '''
    if noise_stats is None:
        noise_stats = get_noise_stats(df, winsize)
    mean_ax = noise_stats[0][0][0]
    mean_ay = noise_stats[0][0][1]
    mean_az = noise_stats[0][0][2]
    mean_gx = noise_stats[0][1][0]
    mean_gy = noise_stats[0][1][1]
    mean_gz = noise_stats[0][1][2]
    std_ax = noise_stats[1][0][0]
    std_ay = noise_stats[1][0][1]
    std_az = noise_stats[1][0][2]
    std_gx = noise_stats[1][1][0]
    std_gy = noise_stats[1][1][1]
    std_gz = noise_stats[1][1][2]
    if reduce_noise:
        std_ax /= reduce_factor
        std_ay /= reduce_factor
        std_az /= reduce_factor
        std_gx /= reduce_factor
        std_gy /= reduce_factor
        std_gz /= reduce_factor

#     print(num_noise)
#     print(len(df.columns))
    zers = pd.DataFrame(np.zeros((num_noise, len(df.columns))), columns=df.columns)
    df = pd.concat([zers, df.loc[:]], axis=0).reset_index(drop=True)

    df.loc[:num_noise-1,'acc_x'] = np.random.normal(loc=mean_ax, scale=std_ax, size=num_noise)
    df.loc[:num_noise-1,'acc_y'] = np.random.normal(loc=mean_ay, scale=std_ay, size=num_noise)
    df.loc[:num_noise-1,'acc_z'] = np.random.normal(loc=mean_az, scale=std_az, size=num_noise)
    df.loc[:num_noise-1,'gyr_x'] = np.random.normal(loc=mean_gx, scale=std_gx, size=num_noise)
    df.loc[:num_noise-1,'gyr_y'] = np.random.normal(loc=mean_gy, scale=std_gy, size=num_noise)
    df.loc[:num_noise-1,'gyr_z'] = np.random.normal(loc=mean_gz, scale=std_gz, size=num_noise)

    return df


def madgwick_eulers_mag(data, sample_rate, add_zeros=2000,  gain=0.5, acc_rej=10, mag_rej=20, sec=5):
    '''
    the function calculates Euler angles from IMU measurements using Madgwick magnetometer-enabled algorithm
    data: IMU measurements in Pandas DataFrame table,
          gyroscope measurements columns should contain 'gyr' in their name,
          acceleromenter measurements columns should contain 'acc' in their name,
          magnetometer measurements columns should contain 'mag' in their name
    sample_rate: Hertz, sampling rate of IMU data
    add_zeros: add normally distributed noise in the beginning of a dataframe (for better convergence)
    gain, acc_rej, mag_rej, sec: Madgwick algorithm parameters

    Returns: Euler angles (X, Y, Z)
    '''
    df = data.copy()
    zers = pd.DataFrame(np.random.normal(loc=0, scale=0.01, size=(add_zeros, len(df.columns))), columns=df.columns)
    zers['acc_y'] +=1
    df = pd.concat([zers, df.loc[:]], axis=0).reset_index(drop=True)
    timestamp = np.array(df['seconds'])
    gyroscope = np.array(df[df.columns[df.columns.str.contains('gyr')]])
    accelerometer = np.array(df[df.columns[df.columns.str.contains('acc')]])
    magnetometer = np.array(df[df.columns[df.columns.str.contains('mag')]])
#     offset = imufusion.Offset(sample_rate)
    ahrs_ = imufusion.Ahrs()
    ahrs_.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                   gain,  # gain
                                   acc_rej,  # acceleration rejection
                                   mag_rej,  # magnetic rejection
                                    round(sec * sample_rate))
    delta_time = np.diff(timestamp, prepend=timestamp[0])
    euler = np.empty((len(timestamp), 3))
    for index in range(len(timestamp)):
#         gyroscope[index] = offset.update(gyroscope[index])
        ahrs_.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])
        euler[index] = ahrs_.quaternion.to_euler()
    return euler[add_zeros:]


def madgwick_eulers_no_mag(data, sample_rate, add_zeros=1000, gain=0.5, acc_rej=10, sec=5):
    '''
    the function calculates Euler angles from IMU measurements using Madgwick magnetometer-disabled algorithm
    data: IMU measurements in Pandas DataFrame table,
          gyroscope measurements columns should contain 'gyr' in their name,
          acceleromenter measurements columns should contain 'acc' in their name,
    sample_rate: Hertz, sampling rate of IMU data
    add_zeros: add normally distributed noise in the beginning of a dataframe (for better convergence)
    gain, acc_rej, sec: Madgwick algorithm parameters

    Returns: Euler angles (X, Y, Z)
    '''
    df = data.copy()
    zers = pd.DataFrame(np.random.normal(loc=0, scale=0.01, size=(add_zeros, len(df.columns))), columns=df.columns)
    zers['acc_y'] +=1
    df = pd.concat([zers, df.loc[:]], axis=0).reset_index(drop=True)
    timestamp = np.array(df['seconds'])
    gyroscope = np.array(df[df.columns[df.columns.str.contains('gyr')]])
    accelerometer = np.array(df[df.columns[df.columns.str.contains('acc')]])
    offset = imufusion.Offset(sample_rate)
    ahrs_ = imufusion.Ahrs()
    euler = np.empty((len(timestamp), 3))
    ahrs_.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                    gain,  # gain
                                   acc_rej,  # acceleration rejection
                                   20,  # magnetic rejection - fixed
                                    round(sec * sample_rate))
    for index in range(len(timestamp)):
#         gyroscope[index] = offset.update(gyroscope[index])
        ahrs_.update_no_magnetometer(gyroscope[index,], accelerometer[index], 1 / sample_rate)
        euler[index] = ahrs_.quaternion.to_euler()
    return euler[add_zeros:]


def thr_madgwick_eulers(data, sample_rate, noise_stats=None, zeros=False, winsize=50, add_zeros=2000, thr=0.1, gain=0.5, acc_rej=10, sec=5,
                              only_below_zero=False, return_full_data=False, reduce_noise=False, reduce_factor=2):
    '''
    Modified "thresholded" version of a Madgwick algorithm
    noise_stats: statistics of the noise - mean and variance
    zeros: False - add noise at the beginning of data, for Madgwick to converge, True - add zeros at the beginning
    add_zeros: number of zeros/noise points to add at the beginning
    thr: threshold (norm of the gyroscope vector)
    winsize: window size to calculate signal properties (mean and variance)
    only_below_zero: if True - cut all the points above zero (was useful for my data)
    return_full_data: if True - return data with added noise / zeros
    reduce_noise: if True - reduce noise variance (from the variance obtained from the actual data)
    reduce_factor: factor to reduce it by


    Returns: Euler angles
    '''
    counter = 0
    df = data.copy()
    if zeros: # add zeros
        zers = pd.DataFrame(np.zeros((add_zeros, len(df.columns))), columns=df.columns)
        zers['acc_y'] +=1
        df_noise = pd.concat([zers,df],axis=0).reset_index(drop=True)
    else: # add noise

        if noise_stats is None:
            noise_stats = get_noise_stats(df, winsize=winsize)
        df_noise = add_noise(df, winsize=50, noise_stats=noise_stats, num_noise=add_zeros, reduce_noise=reduce_noise)

    timestamp = np.array(df_noise['seconds'])
    gyroscope = np.array(df_noise[sorted(df_noise.columns[df_noise.columns.str.contains('gyr')])])
    accelerometer = np.array(df_noise[sorted(df_noise.columns[df_noise.columns.str.contains('acc')])])

    ahrs_ = imufusion.Ahrs()
    below_sec = []
    below_eul = np.empty(shape=[0,3])
    euler = np.empty((len(timestamp), 3))
    ahrs_.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                        gain,  # gain
                                   acc_rej,  # acceleration rejection
                                   20,  # magnetic rejection
                                    round(sec * sample_rate))
    for index in range(len(timestamp)):
        if (np.linalg.norm(gyroscope[index])>=thr):
            ahrs_.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / sample_rate)
            euler[index] = ahrs_.quaternion.to_euler()
        else:
            ahrs_.update_no_magnetometer(np.array([0.,0.,0.]), np.array([0.,0.,0.]), 1 / sample_rate)
            euler[index] = ahrs_.quaternion.to_euler()
            below_sec.append(timestamp[index])
            below_eul = np.append(below_eul, [euler[index]], axis=0)
            counter += 1
        if only_below_zero:
            if euler[index,2] > 0:
                euler[index,2] = 0
    if return_full_data:
        return df_noise, euler
    else:
        return euler[add_zeros:]