import numpy as np
from pyquaternion import Quaternion
import broad.example_code.broad_utils as bu
from collections import deque

class MadgDotFilter:
    def __init__(self, gyr:np.ndarray,acc:np.ndarray,mag:np.ndarray,delta=0.01,beta=0.01,q0=np.array([1,0,0,0])):
        
        assert gyr.shape[0] == acc.shape[0]
        assert gyr.shape[0] == mag.shape[0]
        
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.beta = beta
        self.delta = delta
        self.q0 = Quaternion(q0)
        
        self.Q = []
    
    def _removeBias(self,n=1000):
        gyr_bias = np.mean(self.gyr[0:n], axis=0)  
        self.gyr = self.gyr - gyr_bias
    def filter(self):
        
        
        q_prev = self.q0
        self._removeBias()
        
        q_avg =bu.quatFromAccMag(np.mean(self.acc[0:1000], axis=0),np.mean(self.mag[0:1000], axis=0))
        q_avg = Quaternion(q_avg)

        
        
        for i in range(self.gyr.shape[0]):
            g = self.gyr[i]
            a = self.acc[i]
            m = self.mag[i]
            
            q_sensor = bu.quatFromAccMag(a,m)
            q_sensor = Quaternion(q_sensor)*q_avg.inverse
            
            gyr_quat = Quaternion(0,g[0],g[1],g[2])
            
            loss_dotprod = np.dot(q_sensor.elements,q_prev.elements)
            loss = 1.0 - loss_dotprod**2
            jacobian = np.array([-q_sensor.w,-q_sensor.x,-q_sensor.y,-q_sensor.z])*loss_dotprod*(2)
            
            q_jacob = Quaternion(loss * jacobian)
            q_dot = 0.5 * q_prev * gyr_quat - (self.beta) * q_jacob
            
            q_prev = (q_prev + q_dot*self.delta).normalised

            self.Q.append(q_prev.elements)

        self.Q = np.array(self.Q)


class MadgwickAHRSAB:
    def __init__(self, sample_period=1/256, beta=0.1,q0=Quaternion()):
        self.sample_period = sample_period
        self.beta = beta
        self.q = q0  # identity quaternion
        
        self.AB_window = deque([np.zeros(4)]*10,maxlen=10)
        
        self.coefficients=np.array(
            [
                [1, 0, 0, 0, 0],
                [3/2,-1/2, 0, 0, 0],
                [23/12,-16/12,5/12,0,0],
                [55/24,-59/24,37/24,-9/24,0],
                [1901/720,-2774/720,2616/720,-1274/720,251/720],
            ],dtype=np.float32
        )
        
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

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz,degree=5):
        q1, q2, q3, q4 = self.q.elements

        # Normalize accelerometer
        norm_acc = np.linalg.norm([ax, ay, az])
        if norm_acc == 0:
            return
        ax, ay, az = ax / norm_acc, ay / norm_acc, az / norm_acc

        # Normalize magnetometer
        norm_mag = np.linalg.norm([mx, my, mz])
        if norm_mag == 0:
            return
        mx, my, mz = mx / norm_mag, my / norm_mag, mz / norm_mag

        # Auxiliary variables to avoid repeated calculations
        _2q1mx = 2.0 * q1 * mx
        _2q1my = 2.0 * q1 * my
        _2q1mz = 2.0 * q1 * mz
        _2q2mx = 2.0 * q2 * mx
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4
        _2q1q3 = 2.0 * q1 * q3
        _2q3q4 = 2.0 * q3 * q4
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        # Reference direction of Earth's magnetic field
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        # Gradient descent algorithm corrective step
        s1 = -_2q3 * (2.0 * q2q4 - _2q1q3 - ax) + _2q2 * (2.0 * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s2 = _2q4 * (2.0 * q2q4 - _2q1q3 - ax) + _2q1 * (2.0 * q1q2 + _2q3q4 - ay) - 4.0 * q2 * (1 - 2.0 * q2q2 - 2.0 * q3q3 - az) + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s3 = -_2q1 * (2.0 * q2q4 - _2q1q3 - ax) + _2q4 * (2.0 * q1q2 + _2q3q4 - ay) - 4.0 * q3 * (1 - 2.0 * q2q2 - 2.0 * q3q3 - az) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s4 = _2q2 * (2.0 * q2q4 - _2q1q3 - ax) + _2q3 * (2.0 * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        norm_s = np.linalg.norm([s1, s2, s3, s4])
        if norm_s == 0:
            return
        s1, s2, s3, s4 = s1 / norm_s, s2 / norm_s, s3 / norm_s, s4 / norm_s

        # Rate of change of quaternion from gyroscope
        q_dot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        q_dot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        q_dot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        q_dot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4
        
        self.AB_window.appendleft(np.array([q_dot1, q_dot2, q_dot3, q_dot4]))
        
        ab_0 = self.AB_window[0]
        ab_1 = self.AB_window[1]
        ab_2 = self.AB_window[2]
        ab_3 = self.AB_window[3]
        ab_4 = self.AB_window[4]



        # Integrate to yield quaternion
        q1 += (self.coefficients[degree-1,0]*ab_0[0] + self.coefficients[degree-1,1]*ab_1[0] + self.coefficients[degree-1,2]*ab_2[0] + self.coefficients[degree-1,3]*ab_3[0] + self.coefficients[degree-1,4]*ab_4[0]) * (self.sample_period)
        q2 += (self.coefficients[degree-1,0]*ab_0[1] + self.coefficients[degree-1,1]*ab_1[1] + self.coefficients[degree-1,2]*ab_2[1] + self.coefficients[degree-1,3]*ab_3[1] + self.coefficients[degree-1,4]*ab_4[1]) * (self.sample_period)
        q3 += (self.coefficients[degree-1,0]*ab_0[2] + self.coefficients[degree-1,1]*ab_1[2] + self.coefficients[degree-1,2]*ab_2[2] + self.coefficients[degree-1,3]*ab_3[2] + self.coefficients[degree-1,4]*ab_4[2]) * (self.sample_period)
        q4 += (self.coefficients[degree-1,0]*ab_0[3] + self.coefficients[degree-1,1]*ab_1[3] + self.coefficients[degree-1,2]*ab_2[3] + self.coefficients[degree-1,3]*ab_3[3] + self.coefficients[degree-1,4]*ab_4[3]) * (self.sample_period)
        
        q = Quaternion(q1, q2, q3, q4).normalised
        self.q = q

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
