import numpy as np
from pyquaternion import Quaternion
import broad.example_code.broad_utils as bu
from collections import deque


class MadgwickAHRSAB:
    def __init__(self, sample_period=1/256, beta=0.1,q0=Quaternion(),t=0):
        self.sample_period = sample_period
        self.beta = beta
        self.q = q0  # identity quaternion
        self.t=t
        self.AB_window = deque([np.zeros(4)]*10,maxlen=10)
        
        
        self.coefficients = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Order 1
            [3/2, -1/2, 0, 0, 0, 0, 0, 0, 0, 0],  # Order 2
            [23/12, -16/12, 5/12, 0, 0, 0, 0, 0, 0, 0],  # Order 3
            [55/24, -59/24, 37/24, -9/24, 0, 0, 0, 0, 0, 0],  # Order 4
            [1901/720, -2774/720, 2616/720, -1274/720, 251/720, 0, 0, 0, 0, 0],  # Order 5
            [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440, 0, 0, 0, 0],  # Order 6
            [198721/60480, -447288/60480, 705549/60480, -688256/60480, 407139/60480, -134472/60480, 19087/60480, 0, 0, 0],  # Order 7
            [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960, 0, 0],  # Order 8
            [14097247/3628800, -43125206/3628800, 95476786/3628800, -139855262/3628800, 137968480/3628800, -91172642/3628800, 38833486/3628800, -9664106/3628800, 1070017/3628800, 0],  # Order 9
            [30277247/7983360, -104995189/7983360, 265932680/7983360, -454661776/7983360, 538363838/7983360, -444772162/7983360, 252618224/7983360, -94307320/7983360, 20058768/7983360, -1936319/7983360]  # Order 10
        ])


    def updateMARG(self, gyr,acc,mag,degree=5):
        q = self.q
        dt = self.sample_period
        if np.linalg.norm(gyr) == 0:
            return 
        if np.linalg.norm(mag) == 0:
            return 
        
        qDot = (0.5 * q*Quaternion([0, *gyr])).elements                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            a = acc/a_norm
            m = mag/np.linalg.norm(mag)
            # Rotate normalized magnetometer measurements
            q_m = Quaternion([0, *m])
            h = (q*(q_m*(q.conjugate))).elements                      # (eq. 45)
            bx = np.linalg.norm([h[1], h[2]])                       # (eq. 46)
            bz = h[3]
            qw, qx, qy, qz = q.normalised.elements
            # Objective function (eq. 31)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2],
                          2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                          2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                          2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]])
            if np.linalg.norm(f) > 0:
                # Jacobian (eq. 32)
                J = np.array([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                              [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                              [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                              [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                              [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                              [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx          ]])
                gradient = J.T@f                                    # (eq. 34)
                gradient /= np.linalg.norm(gradient)
                qDot -= self.beta*gradient                          # (eq. 33)
   
        
        self.AB_window.appendleft(qDot)
        ab_window_arr = np.array(self.AB_window)   
        update_factor = (self.coefficients[degree-1].reshape((1,10)) @ ab_window_arr).reshape((4,))
        
        q_new = q.elements + (update_factor)*dt   # (eq. 13)
        q_new /= np.linalg.norm(q_new)
        
        self.q = Quaternion(q_new)

    def get_quaternion(self):
        return self.q
    



    def filter_batch(self, gyro_data, accel_data, mag_data,degree=5):
        """
        Applies Madgwick filter to a batch of sensor data.

        Parameters:
            gyro_data: Nx3 array of gyroscope readings in rad/s
            accel_data: Nx3 array of accelerometer readings
            mag_data: Nx3 array of magnetometer readings

        Returns:
            List of Quaternion objects representing orientation at each step
        """
        # gyr_bias = np.mean(gyro_data[0:1000], axis=0)  
        # gyro_data = gyro_data - gyr_bias
        
        assert len(gyro_data) == len(accel_data) == len(mag_data)
        quaternions = []


        for g, a, m in zip(gyro_data, accel_data, mag_data):
            # self.update(*g, *a, *m,degree=degree)
            self.updateMARG(g, a, m,degree=degree)
            quaternions.append(self.get_quaternion().elements)

        return np.array(quaternions)
