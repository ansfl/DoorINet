# Dataset

Training and testing datasets were recorded using several lab doors of Autonomous Navigation and Sensor Fusion Lab in the University of Haifa over three recording sessions.

XDOT IMU sensors were placed on a lab door to measure the door movements. Three different positions of IMUs were explored: next to the door handle, in the middle of the door and next to the door hinge.

Setups are presented in photos below. Setups 1 and 2 are similar to each other, except that Setup 1 used XDOT IMU #14, and Setup 2 used XDOT IMU #8 and also different positions on the door.

<table>
  <tr>
    <td align='center'>Setups 1 and 2</td>
    <td align='center'>Setup 3</td>
  </tr>
  <tr>
    <td> <img src="https://github.com/ansfl/DoorINet/assets/89016122/140b36eb-1e8b-4f70-bb59-53afaa74543a" alt="Setups 1 & 2" width="400" height="300"> </td>
    <td> <img src="https://github.com/ansfl/DoorINet/assets/89016122/1cbebedd-af46-42c2-a821-d0e1338832f8" alt="Setup 3" width="400" height="300"></td>
</tr>
</table>

The datasets are written to CSV **_csv_** files and can be downloaded from [Google drive](https://drive.google.com/drive/folders/11T_5DR6Rnr8eeFteVZ_w3SqR17zyQix1?usp=sharing)

Table structure is as follows:

* **train_dataset.xlsx** contains all the training IMU data used in the paper;
* **unused_data.xlsx** contains IMU data that was recorded but was not used in training for various reasons;
* **12_test.xlsx** is a test dataset recorded by XDOT IMU #12;
* **14_test.xlsx** is a test dataset recorded by XDOT IMU #14;
* **5_test.xlsx** is a test dataset recorded by XDOT IMU #5;

The table structure is as follows:

* _sampletimefine_ - internal time from the IMU, in microseconds;
* _acc_x_ - accelerometer measurements along the X axis, in g;
* _acc_y_ - accelerometer measurements along the Y axis, in g;
* _acc_z_ - accelerometer measurements along the Z axis, in g;
* _gyr_x_ - gyroscope measurements along the X axis, in degrees per second;
* _gyr_y_ - gyroscope measurements along the Y axis, in degrees per second;
* _gyr_z_ - gyroscope measurements along the Z axis, in degrees per second;
* _mag_x_ - magnetometer measurements along the X axis, in nano Tesla;
* _mag_y_ - magnetometer measurements along the Y axis, in nano Tesla;
* _mag_z_ - magnetometer measurements along the Z axis, in nano Tesla;
* _heading_angle_ - resulting heading angle of the door opening (ground truth), in degrees;
* _session_ - recording session number, either 1, 2 or 3;
* _xdot_label_ - label of the XDOT IMU, we had XDOT IMUs labelled 5, 6, 7, 8, 9, 10, 11, 12, 14 and 15 (see photos above);
* _xdot_position_ - position of the XDOT IMU on the door surface, can be either "handle" (close to the door handle), "median" (in the middle of the door) and "hinge" (close to the door hinge);
* _description_ - some textual description of the current experiment
