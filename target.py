from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib.patches import Circle
# %matplotlib inline
import numpy as np
from math import atan2
import matplotlib.pyplot as plt


class Target(object):
    def __init__(self,arenaSize,targetMotConstraints):
        self.setMachineParameters(targetMotConstraints)
        self.point2CV_MotionTable = []
        self.CV2CV_MotionTable = []


    def setMachineParameters(self, machineParams):
        # Use this function to set the machine parameters, will raise an error if list length is 1 or 2
        if len(machineParams)<6 and len(machineParams)>0:
            raise Exception('Not enough arguments in list. Must pass a list with Vmax, Amax, Jmax, and Sampling Frequency')
        if len(machineParams)==6:
            self.Vmax = float(machineParams[0])
            self.Amax = float(machineParams[1])
            self.Jmax = float(machineParams[2])
            self.bearingRange = float(machineParams[3])
            self.bearingRate = float(machineParams[4])
            self.sampFreq = float(machineParams[5])
            self.settlingTime = 0
    
    def p2pMotionFinalTime(self,xStart, xStop, cycleTime = 0, delayTime = 0):
		# This function will generate trajectories for a point to point motion
		## fs			= Sampling frequency, will effect final time
		## xStart 		= Start position
		## xStop 		= End position
		## cycleTime	= If provided, will give the time at which the move should end
		## delayTime 	= If provided, will add a delay to the calculated time

		# This function will return the motion time

        V = self.Vmax
        A = self.Amax
        J = self.Jmax
        fs = self.sampFreq

        # First, lets determine if the Amax value needs to be adjusted based on the values of Vmax and Jmax
        if V * J < A ** 2:
            A = np.sqrt(V * J)

        # Lets determine the direction and distance of the move
        direction = np.sign(xStop - xStart)
        pT = np.abs(xStop - xStart)

        # Now lets calculate the shortest distance required for Amax to be reached, as well as the time
        pAJ = (2 * A ** 3) / (J ** 2)
        TAJ = (4 * A) / J

        # Now lets calculate the shortest distance required for Vmax to be reached, as well as the time
        pVAJ = V ** 2 / A + (A * V) / J
        TVAJ = (2 * A) / J + (2 * V) / A

        # Depending on the total distance of the move, Amax or Vmax may not be reached, which effects the number of
        # motion segments there are. Below we determine the time at which each motion segment begins

        if pT >= pVAJ:  # Seven Segments
            # t1 = A / J
            # t2 = V / A
            # t3 = A / J + V / A
            # t4 = pT / V
            # t5 = A / J + pT / V
            # t6 = V / A + pT / V
            T = A / J + V / A + pT / V

        elif pT > pAJ:  # Five Segments
            # t1 = A / J
            # t2 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A) - A / (2 * J)
            # t3 = (3 * A) / (2 * J) + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A)
            # t4 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A
            T = A / J + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A

        else:
            T = ((32 * pT) / J) ** (1 / 3)
            # t1 = T / 4
            # t2 = (3 * T) / 4

        # dt = 1 / fs  # Sampling interval

        # Now lets determine the final time
        if cycleTime - self.settlingTime > T:
            # The motion should be scaled to finish in the requested cycle time
            # Will need to recalculate the times with the new scaled V, A, and J

            # First, lets scale V, A, and J
            scaleFactor = T / (cycleTime - self.settlingTime)
            V = V * scaleFactor
            A = A * scaleFactor ** 2
            J = J * scaleFactor ** 3

            # Now lets calculate the shortest distance required for Amax to be reached, as well as the time
            pAJ = (2 * A ** 3) / (J ** 2)
            TAJ = (4 * A) / J

            # Now lets calculate the shortest distance required for Vmax to be reached, as well as the time
            pVAJ = V ** 2 / A + (A * V) / J
            TVAJ = (2 * A) / J + (2 * V) / A

            # Depending on the total distance of the move, Amax or Vmax may not be reached, which effects the number of
            # motion segments there are. Below we determine the time at which each motion segment begins

            if pT >= pVAJ:  # Seven Segments
                t1 = A / J
                t2 = V / A
                t3 = A / J + V / A
                t4 = pT / V
                t5 = A / J + pT / V
                t6 = V / A + pT / V
                T = A / J + V / A + pT / V

            elif pT > pAJ:  # Five Segments
                t1 = A / J
                t2 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A) - A / (2 * J)
                t3 = (3 * A) / (2 * J) + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A)
                t4 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A
                T = A / J + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A

            else:
                T = ((32 * pT) / J) ** (1 / 3)
                t1 = T / 4
                t2 = (3 * T) / 4
            Tf = T + self.settlingTime
        # If there is a delay, add that to the final time

        else:
            Tf = T + self.settlingTime


        return Tf

    def p2pMotionTrajectoryGenerator(self, xStart, xStop, cycleTime = 0, delayTime = 0):
		# This function will generate trajectories for a point to point motion
		## fs			= Sampling frequency, will effect final time
		## xStart 		= Start position
		## xStop 		= End position
		## cycleTime	= If provided, will give the time at which the move should end
		## delayTime 	= If provided, will add a delay to the calculated time

		# This function will return the motion time, as well as the trajectory of pos, vel, acc, and jerk

        V = self.Vmax
        A = self.Amax
        J = self.Jmax
        fs = self.sampFreq

        # First, lets make sure the start and stop positions are different
        if xStart == xStop:
            # If they are the same, the axis doesn't move, so for the cycleTime duration, the axis stays at xStart
            if cycleTime==0:
                return 0, [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
            else:
                return 0, [np.array([[0,fs*np.floor(cycleTime/fs)]]), np.array([[xStart,xStart]]), np.array([[0,0]]), np.array([[0,0]]), np.array([[0,0]])]


        # First, lets determine if the Amax value needs to be adjusted based on the values of Vmax and Jmax
        if V*J < A**2:
            A = np.sqrt(V*J)

        # Lets determine the direction and distance of the move
        direction = np.sign(xStop - xStart)
        pT = np.abs(xStop - xStart)

        # Now lets calculate the shortest distance required for Amax to be reached, as well as the time
        pAJ = (2*A**3)/(J**2)
        TAJ = (4*A)/J

        # Now lets calculate the shortest distance required for Vmax to be reached, as well as the time
        pVAJ = V**2/A + (A*V)/J
        TVAJ = (2*A)/J + (2*V)/A

        # Depending on the total distance of the move, Amax or Vmax may not be reached, which effects the number of
        # motion segments there are. Below we determine the time at which each motion segment begins

        if pT >= pVAJ:	# Seven Segments
            t1 = A/J
            t2 = V/A
            t3 = A/J + V/A
            t4 = pT/V
            t5 = A/J + pT/V
            t6 = V/A + pT/V
            T = A/J + V/A + pT/V

        elif pT>pAJ:	# Five Segments
            t1 = A/J
            t2 = (4*A*pT + A**4/J**2)**(1/2)/(2*A) - A/(2*J)
            t3 = (3 * A) / (2 * J) + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A)
            t4 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A
            T = A / J + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A

        else:
            T = ((32*pT)/J)**(1/3)
            t1 = T/4
            t2 = (3*T)/4

        dt = 1/fs	# Sampling interval


        # Now lets determine the final time

        if cycleTime-self.settlingTime>T:
            # The motion should be scaled to finish in the requested cycle time
            # Will need to recalculate the times with the new scaled V, A, and J

            # First, lets scale V, A, and J
            scaleFactor = T/(cycleTime-self.settlingTime)
            V = V*scaleFactor
            A = A*scaleFactor**2
            J = J*scaleFactor**3

            # Now lets calculate the shortest distance required for Amax to be reached, as well as the time
            pAJ = (2 * A ** 3) / (J ** 2)
            TAJ = (4 * A) / J

            # Now lets calculate the shortest distance required for Vmax to be reached, as well as the time
            pVAJ = V ** 2 / A + (A * V) / J
            TVAJ = (2 * A) / J + (2 * V) / A

            # Depending on the total distance of the move, Amax or Vmax may not be reached, which effects the number of
            # motion segments there are. Below we determine the time at which each motion segment begins

            if pT >= pVAJ:  # Seven Segments
                t1 = A / J
                t2 = V / A
                t3 = A / J + V / A
                t4 = pT / V
                t5 = A / J + pT / V
                t6 = V / A + pT / V
                T = A / J + V / A + pT / V

            elif pT > pAJ:  # Five Segments
                t1 = A / J
                t2 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A) - A / (2 * J)
                t3 = (3 * A) / (2 * J) + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / (2 * A)
                t4 = (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A
                T = A / J + (4 * A * pT + A ** 4 / J ** 2) ** (1 / 2) / A

            else:
                T = ((32 * pT) / J) ** (1 / 3)
                t1 = T / 4
                t2 = (3 * T) / 4
            Tf = T+self.settlingTime
            # If there is a delay, add that to the final time


        else:
            Tf = T+self.settlingTime

        if delayTime > 0:
            Tf = delayTime + Tf

        steps = np.arange(0,T,dt).reshape(-1,1)
        pos = np.zeros((steps.shape[0],1))
        vel = np.zeros((steps.shape[0],1))
        acc = np.zeros((steps.shape[0],1))
        jerk = np.zeros((steps.shape[0],1))

        # Lets fill out the trajectory arrays created above
        if pT>=pVAJ:		# 7 Segments
            # Positions
            pos[np.logical_and(steps<t1,steps>=0)] = (J*steps[np.logical_and(steps<t1,steps>=0)]**3)/6
            pos[np.logical_and(steps<t2,steps>=t1)] = (A*steps[np.logical_and(steps<t2,steps>=t1)]**2)/2 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<t2,steps>=t1)])/(2*J)
            pos[np.logical_and(steps < t3, steps >= t2)] = (A*steps[np.logical_and(steps<t3,steps>=t2)]**2)/2 - (J*steps[np.logical_and(steps<t3,steps>=t2)]**3)/6 + (J*t2**3)/6 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<t3,steps>=t2)])/(2*J) - (J*steps[np.logical_and(steps<t3,steps>=t2)]*t2**2)/2 + (J*steps[np.logical_and(steps<t3,steps>=t2)]**2*t2)/2
            pos[np.logical_and(steps < t4, steps >= t3)] = (J*t2**3)/6 - (A*t3**2)/2 + (J*t3**3)/3 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<t4,steps>=t3)])/(2*J) + A*steps[np.logical_and(steps<t4,steps>=t3)]*t3 - (J*steps[np.logical_and(steps<t4,steps>=t3)]*t2**2)/2 - (J*steps[np.logical_and(steps<t4,steps>=t3)]*t3**2)/2 - (J*t2*t3**2)/2 + J*steps[np.logical_and(steps<t4,steps>=t3)]*t2*t3
            pos[np.logical_and(steps < t5, steps >= t4)] = (J*t2**3)/6 - (J*steps[np.logical_and(steps<t5,steps>=t4)]**3)/6 - (A*t3**2)/2 + (J*t3**3)/3 + (J*t4**3)/6 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<t5,steps>=t4)])/(2*J) + A*steps[np.logical_and(steps<t5,steps>=t4)]*t3 - (J*steps[np.logical_and(steps<t5,steps>=t4)]*t2**2)/2 - (J*steps[np.logical_and(steps<t5,steps>=t4)]*t3**2)/2 - (J*steps[np.logical_and(steps<t5,steps>=t4)]*t4**2)/2 + (J*steps[np.logical_and(steps<t5,steps>=t4)]**2*t4)/2 - (J*t2*t3**2)/2 + J*steps[np.logical_and(steps<t5,steps>=t4)]*t2*t3
            pos[np.logical_and(steps < t6, steps >= t5)] = (J*t2**3)/6 - (A*t3**2)/2 - (A*t5**2)/2 - (A*steps[np.logical_and(steps<t6,steps>=t5)]**2)/2 + (J*t3**3)/3 + (J*t4**3)/6 + (J*t5**3)/3 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<t6,steps>=t5)])/(2*J) + A*steps[np.logical_and(steps<t6,steps>=t5)]*t3 + A*steps[np.logical_and(steps<t6,steps>=t5)]*t5 - (J*steps[np.logical_and(steps<t6,steps>=t5)]*t2**2)/2 - (J*steps[np.logical_and(steps<t6,steps>=t5)]*t3**2)/2 - (J*steps[np.logical_and(steps<t6,steps>=t5)]*t4**2)/2 - (J*steps[np.logical_and(steps<t6,steps>=t5)]*t5**2)/2 - (J*t2*t3**2)/2 - (J*t4*t5**2)/2 + J*steps[np.logical_and(steps<t6,steps>=t5)]*t2*t3 + J*steps[np.logical_and(steps<t6,steps>=t5)]*t4*t5
            pos[np.logical_and(steps < T, steps >= t6)] = (J*steps[np.logical_and(steps<T,steps>=t6)]**3)/6 - (A*t3**2)/2 - (A*t5**2)/2 - (A*steps[np.logical_and(steps<T,steps>=t6)]**2)/2 + (J*t2**3)/6 + (J*t3**3)/3 + (J*t4**3)/6 + (J*t5**3)/3 - (J*t6**3)/6 + A**3/(6*J**2) - (A**2*steps[np.logical_and(steps<T,steps>=t6)])/(2*J) + A*steps[np.logical_and(steps<T,steps>=t6)]*t3 + A*steps[np.logical_and(steps<T,steps>=t6)]*t5 - (J*steps[np.logical_and(steps<T,steps>=t6)]*t2**2)/2 - (J*steps[np.logical_and(steps<T,steps>=t6)]*t3**2)/2 - (J*steps[np.logical_and(steps<T,steps>=t6)]*t4**2)/2 - (J*steps[np.logical_and(steps<T,steps>=t6)]*t5**2)/2 - (J*t2*t3**2)/2 + (J*steps[np.logical_and(steps<T,steps>=t6)]*t6**2)/2 - (J*steps[np.logical_and(steps<T,steps>=t6)]**2*t6)/2 - (J*t4*t5**2)/2 + J*steps[np.logical_and(steps<T,steps>=t6)]*t2*t3 + J*steps[np.logical_and(steps<T,steps>=t6)]*t4*t5
            pos[steps >= T] =(J*T**3)/6 - (A*t3**2)/2 - (A*t5**2)/2 - (A*T**2)/2 + (J*t2**3)/6 + (J*t3**3)/3 + (J*t4**3)/6 + (J*t5**3)/3 - (J*t6**3)/6 + A**3/(6*J**2) - (A**2*T)/(2*J) + A*T*t3 + A*T*t5 - (J*T*t2**2)/2 - (J*T*t3**2)/2 - (J*T*t4**2)/2 - (J*T*t5**2)/2 - (J*t2*t3**2)/2 + (J*T*t6**2)/2 - (J*T**2*t6)/2 - (J*t4*t5**2)/2 + J*T*t2*t3 + J*T*t4*t5

            # Velocities
            vel[np.logical_and(steps<t1,steps>=0)] = (J*steps[np.logical_and(steps<t1,steps>=0)]**2)/2
            vel[np.logical_and(steps<t2,steps>=t1)] = -A**2/(2*J) + steps[np.logical_and(steps<t2,steps>=t1)]*A
            vel[np.logical_and(steps < t3, steps >= t2)] = - A**2/(2*J) + steps[np.logical_and(steps<t3,steps>=t2)]*A - (J*steps[np.logical_and(steps<t3,steps>=t2)]**2)/2 - (J*t2**2)/2 + J*steps[np.logical_and(steps<t3,steps>=t2)]*t2
            vel[np.logical_and(steps < t4, steps >= t3)] =- A**2/(2*J) + t3*A - (J*t2**2)/2 - (J*t3**2)/2 + J*t2*t3
            vel[np.logical_and(steps < t5, steps >= t4)] =- A**2/(2*J) + t3*A - (J*steps[np.logical_and(steps<t5,steps>=t4)]**2)/2 - (J*t2**2)/2 - (J*t3**2)/2 - (J*t4**2)/2 + J*steps[np.logical_and(steps<t5,steps>=t4)]*t4 + J*t2*t3;
            vel[np.logical_and(steps < t6, steps >= t5)] =A*t3 - A*steps[np.logical_and(steps<t6,steps>=t5)] + A*t5 - (J*t2**2)/2 - (J*t3**2)/2 - (J*t4**2)/2 - (J*t5**2)/2 - A**2/(2*J) + J*t2*t3 + J*t4*t5
            vel[np.logical_and(steps < T, steps >= t6)] =A*t3 - A*steps[np.logical_and(steps<T,steps>=t6)] + A*t5 + (J*steps[np.logical_and(steps<T,steps>=t6)]**2)/2 - (J*t2**2)/2 - (J*t3**2)/2 - (J*t4**2)/2 - (J*t5**2)/2 + (J*t6**2)/2 - A**2/(2*J) + J*t2*t3 - J*steps[np.logical_and(steps<T,steps>=t6)]*t6 + J*t4*t5
            vel[steps >= T] = 0

            # Accelerations
            acc[np.logical_and(steps<t1,steps>=0)] = J*steps[np.logical_and(steps<t1,steps>=0)]
            acc[np.logical_and(steps<t2,steps>=t1)] = A
            acc[np.logical_and(steps < t3, steps >= t2)] =A - J*steps[np.logical_and(steps<t3,steps>=t2)] + J*t2
            acc[np.logical_and(steps < t4, steps >= t3)] =0
            acc[np.logical_and(steps < t5, steps >= t4)] =J*t4 - J*steps[np.logical_and(steps<t5,steps>=t4)]
            acc[np.logical_and(steps < t6, steps >= t5)] = -A
            acc[np.logical_and(steps < T, steps >= t6)] = J*steps[np.logical_and(steps<T,steps>=t6)] - A - J*t6
            acc[steps >= T] = 0

            # Jerks
            jerk[np.logical_and(steps < t1, steps >= 0)] = J
            jerk[np.logical_and(steps < t2, steps >= t1)] = 0
            jerk[np.logical_and(steps < t3, steps >= t2)] = -J
            jerk[np.logical_and(steps < t4, steps >= t3)] = 0
            jerk[np.logical_and(steps < t5, steps >= t4)] = -J
            jerk[np.logical_and(steps < t6, steps >= t5)] = 0
            jerk[np.logical_and(steps < T, steps >= t6)] = J
            jerk[steps >= T] = 0
        elif pT>pAJ:		# 5 Segments
            # Positions
            pos[np.logical_and(steps<t1,steps>=0)] =(J*steps[np.logical_and(steps<t1,steps>=0)]**3)/6
            pos[np.logical_and(steps<t2,steps>=t1)] =(A*steps[np.logical_and(steps<t2,steps>=t1)]**2)/2 + (J*steps[np.logical_and(steps<t2,steps>=t1)]*t1**2)/2 - A*steps[np.logical_and(steps<t2,steps>=t1)]*t1 - (J*t1**3)/3 + (A*t1**2)/2
            pos[np.logical_and(steps < t3, steps >= t2)] =- (J*steps[np.logical_and(steps<t3,steps>=t2)]**3)/6 + (J*steps[np.logical_and(steps<t3,steps>=t2)]**2*t2)/2 + (A*steps[np.logical_and(steps<t3,steps>=t2)]**2)/2 + (J*steps[np.logical_and(steps<t3,steps>=t2)]*t1**2)/2 - A*steps[np.logical_and(steps<t3,steps>=t2)]*t1 - (J*steps[np.logical_and(steps<t3,steps>=t2)]*t2**2)/2 - (J*t1**3)/3 + (A*t1**2)/2 + (J*t2**3)/6
            pos[np.logical_and(steps < t4, steps >= t3)] =(A*t1**2)/2 - (A*steps[np.logical_and(steps<t4,steps>=t3)]**2)/2 - A*t3**2 - (J*t1**3)/3 + (J*t2**3)/6 + (J*t3**3)/3 - A*steps[np.logical_and(steps<t4,steps>=t3)]*t1 + 2*A*steps[np.logical_and(steps<t4,steps>=t3)]*t3 + (J*steps[np.logical_and(steps<t4,steps>=t3)]*t1**2)/2 - (J*steps[np.logical_and(steps<t4,steps>=t3)]*t2**2)/2 - (J*steps[np.logical_and(steps<t4,steps>=t3)]*t3**2)/2 - (J*t2*t3**2)/2 + J*steps[np.logical_and(steps<t4,steps>=t3)]*t2*t3
            pos[np.logical_and(steps < T, steps >= t4)] =(J*steps[np.logical_and(steps<T,steps>=t4)]**3)/6 - (J*steps[np.logical_and(steps<T,steps>=t4)]**2*t4)/2 - (A*steps[np.logical_and(steps<T,steps>=t4)]**2)/2 + (J*steps[np.logical_and(steps<T,steps>=t4)]*t1**2)/2 - A*steps[np.logical_and(steps<T,steps>=t4)]*t1 - (J*steps[np.logical_and(steps<T,steps>=t4)]*t2**2)/2 + J*steps[np.logical_and(steps<T,steps>=t4)]*t2*t3 - (J*steps[np.logical_and(steps<T,steps>=t4)]*t3**2)/2 + 2*A*steps[np.logical_and(steps<T,steps>=t4)]*t3 + (J*steps[np.logical_and(steps<T,steps>=t4)]*t4**2)/2 - (J*t1**3)/3 + (A*t1**2)/2 + (J*t2**3)/6 - (J*t2*t3**2)/2 + (J*t3**3)/3 - A*t3**2 - (J*t4**3)/6
            pos[steps >= T] =(J*T**3)/6 - (J*T**2*t4)/2 - (A*T**2)/2 + (J*T*t1**2)/2 - A*T*t1 - (J*T*t2**2)/2 + J*T*t2*t3 - (J*T*t3**2)/2 + 2*A*T*t3 + (J*T*t4**2)/2 - (J*t1**3)/3 + (A*t1**2)/2 + (J*t2**3)/6 - (J*t2*t3**2)/2 + (J*t3**3)/3 - A*t3**2 - (J*t4**3)/6

            # Velocities
            vel[np.logical_and(steps < t1, steps >= 0)] =(J*steps[np.logical_and(steps<t1,steps>=0)]**2)/2
            vel[np.logical_and(steps < t2, steps >= t1)] =(J*t1**2)/2 - A*t1 + A*steps[np.logical_and(steps<t2,steps>=t1)]
            vel[np.logical_and(steps < t3, steps >= t2)] =(J*t1**2)/2 - A*t1 + A*steps[np.logical_and(steps<t3,steps>=t2)] - (J*steps[np.logical_and(steps<t3,steps>=t2)]**2)/2 - (J*t2**2)/2 + J*steps[np.logical_and(steps<t3,steps>=t2)]*t2
            vel[np.logical_and(steps < t4, steps >= t3)] =(J*t1**2)/2 - A*t1 - A*steps[np.logical_and(steps<t4,steps>=t3)] + 2*A*t3 - (J*t2**2)/2 - (J*t3**2)/2 + J*t2*t3
            vel[np.logical_and(steps < T, steps >= t4)] =(J*steps[np.logical_and(steps<T,steps>=t4)]**2)/2 - J*steps[np.logical_and(steps<T,steps>=t4)]*t4 - A*steps[np.logical_and(steps<T,steps>=t4)] + (J*t1**2)/2 - A*t1 + (J*t4**2)/2 + 2*A*t3 - (J*t2**2)/2 - (J*t3**2)/2 + J*t2*t3
            vel[steps >= T] = 0

            # Accelerations
            acc[np.logical_and(steps < t1, steps >= 0)] =J*steps[np.logical_and(steps<t1,steps>=0)]
            acc[np.logical_and(steps < t2, steps >= t1)] =A
            acc[np.logical_and(steps < t3, steps >= t2)] =A - J*steps[np.logical_and(steps<t3,steps>=t2)] + J*t2
            acc[np.logical_and(steps < t4, steps >= t3)] =-A
            acc[np.logical_and(steps < T, steps >= t4)] = J*steps[np.logical_and(steps<T,steps>=t4)] - A - J*t4
            acc[steps >= T] = 0

            # Jerks
            jerk[np.logical_and(steps < t1, steps >= 0)] = J
            jerk[np.logical_and(steps < t2, steps >= t1)] = 0
            jerk[np.logical_and(steps < t3, steps >= t2)] =-J
            jerk[np.logical_and(steps < t4, steps >= t3)] =0
            jerk[np.logical_and(steps < T, steps >= t4)] =J
            jerk[steps >= T] =0

        else:		# 3 Segments
            # Positions
            pos[np.logical_and(steps<t1,steps>=0)] =(J*steps[np.logical_and(steps<t1,steps>=0)]**3)/6
            pos[np.logical_and(steps<t2,steps>=t1)] =(J*T**3)/192 - (J*T**2*steps[np.logical_and(steps<t2,steps>=t1)])/16 + (J*T*steps[np.logical_and(steps<t2,steps>=t1)]**2)/4 - (J*steps[np.logical_and(steps<t2,steps>=t1)]**3)/6
            pos[np.logical_and(steps < T, steps >= t2)] =- (13*J*T**3)/96 + (J*T**2*steps[np.logical_and(steps<T,steps>=t2)])/2 - (J*T*steps[np.logical_and(steps<T,steps>=t2)]**2)/2 + (J*steps[np.logical_and(steps<T,steps>=t2)]**3)/6
            pos[steps >= T] =- (13*J*T**3)/96 + (J*T**2*T)/2 - (J*T*T**2)/2 + (J*T**3)/6

            # Velocities
            vel[np.logical_and(steps < t1, steps >= 0)] =(J*steps[np.logical_and(steps<t1,steps>=0)]**2)/2
            vel[np.logical_and(steps < t2, steps >= t1)] =(J*T*steps[np.logical_and(steps<t2,steps>=t1)])/2 - (J*steps[np.logical_and(steps<t2,steps>=t1)]**2)/2 - (J*T**2)/16
            vel[np.logical_and(steps < T, steps >= t2)] =(J*T**2)/2 - J*T*steps[np.logical_and(steps<T,steps>=t2)] + (J*steps[np.logical_and(steps<T,steps>=t2)]**2)/2
            vel[steps >= T] = 0

            # Accelerations
            acc[np.logical_and(steps < t1, steps >= 0)] =J*steps[np.logical_and(steps<t1,steps>=0)]
            acc[np.logical_and(steps < t2, steps >= t1)] =(J*T)/2 - J*steps[np.logical_and(steps<t2,steps>=t1)]
            acc[np.logical_and(steps < T, steps >= t2)] =J*steps[np.logical_and(steps<T,steps>=t2)] - J*T
            acc[steps >= T] = 0

            # Jerks
            jerk[np.logical_and(steps < t1, steps >= 0)] = J
            jerk[np.logical_and(steps < t2, steps >= t1)] =-J
            jerk[np.logical_and(steps < T, steps >= t2)] =J
            jerk[steps >= T] = 0

        # Finally, we need to ensure the direction of motion and start position are correct
        if direction < 0:
            pos = direction*pos+xStart
            vel = direction*vel
            acc = direction*acc
            jerk = direction*jerk
        else:
            pos = pos+xStart
        delaySteps = np.arange(0, delayTime, dt).reshape(-1, 1)
        settlingTimeSteps = np.arange(dt, self.settlingTime+dt, dt).reshape(-1, 1)+delayTime+steps[-1]
        if not np.any(delaySteps):
            delaySteps = np.vstack((delaySteps,np.array([0])))


        steps = np.vstack((delaySteps,steps+delayTime,settlingTimeSteps))
        pos = np.vstack((np.zeros((delaySteps.shape[0], 1))+xStart,pos,np.zeros((settlingTimeSteps.shape[0], 1))+xStop))
        vel = np.vstack((np.zeros((delaySteps.shape[0], 1)),vel,np.zeros((settlingTimeSteps.shape[0], 1))))
        acc = np.vstack((np.zeros((delaySteps.shape[0], 1)),acc,np.zeros((settlingTimeSteps.shape[0], 1))))
        jerk = np.vstack((np.zeros((delaySteps.shape[0], 1)),jerk,np.zeros((settlingTimeSteps.shape[0], 1))))

        return Tf, [pos, vel, acc, jerk, steps]

    def trajectorySelector(self, trajSelection,initialConditions=[],finalConditions=[],simulationTime=30,randomPoints=[]):
        '''
        This function is used to select which trajectory will be used for the
        target.

        trajSelection == Integer
            trajSelection=0 -> Stationary
            trajSelection=1 -> Point to Point
            trajSelection=2 -> Circular
            trajSelection=3 -> Random

        initialConditions == 6x1 Numpy Array
            initialConditions[0] = X Position
            initialConditions[1] = Y Position
            initialConditions[2] = X Velocity
            initialConditions[3] = Y Velocity
            initialConditions[4] = X Acceleration
            initialConditions[5] = Y Acceleration

        finalConditions == 6x1 Numpy Array
            initialConditions[0] = X Position
            initialConditions[1] = Y Position
            initialConditions[2] = X Velocity
            initialConditions[3] = Y Velocity
            initialConditions[4] = X Acceleration
            initialConditions[5] = Y Acceleration

        simulationTime == float, total simulation time

        Function creates a self.trajectory array with shape int(simulationTime/self.sampFreq) x 5
            Column 1 = X Position
            Column 2 = Y Position
            Column 3 = X Velocity
            Column 4 = Y Velocity
            Column 5 = Step Time
        '''
        
        if trajSelection==0:    #Stationary
            self.trajectory = np.zeros((int(simulationTime*self.sampFreq),5))
            self.trajectory[:,0:2] = initialConditions[0:2]

        elif trajSelection==1:  # Point to Point
            # First, need to determine the time needed to get from initial to
            #   final position in both X and Y (p2p method is 1D). Using the
            #   largest time, I set up the trajectories
            xTF = self.p2pMotionFinalTime(initialConditions[0],finalConditions[0])
            yTF = self.p2pMotionFinalTime(initialConditions[1],finalConditions[1])

            actualTF = max(xTF,yTF)

            if actualTF<simulationTime:
                actualTF=simulationTime
            

            TF,xTraj = self.p2pMotionTrajectoryGenerator(initialConditions[0],finalConditions[0],actualTF)
            TF,yTraj = self.p2pMotionTrajectoryGenerator(initialConditions[1],finalConditions[1],actualTF)

            xTraj = np.array(xTraj)
            yTraj = np.array(yTraj)

            if xTraj.shape[0] > yTraj.shape[0]:
                newY = np.ones(xTraj.shape)
                newY[0:yTraj.shape[0]+1,:] = yTraj
                for ii in range(0,yTraj.shape[1]):
                    newY[yTraj.shape[0]+1:,ii]=yTraj[-1,ii]
                yTraj = newY
            elif xTraj.shape[0] < yTraj.shape[0]:
                newX = np.ones(yTraj.shape)
                newX[0:xTraj.shape[0]+1,:] = xTraj
                for ii in range(0,xTraj.shape[1]):
                    newX[xTraj.shape[0]+1:,ii]=xTraj[-1,ii]
                xTraj = newX

        
            # print(xTraj.shape)
            # print(yTraj.shape)

            self.trajectory = np.empty((xTraj[0].shape[0],5))
            self.trajectory[:,0] = xTraj[0].flatten()
            self.trajectory[:,1] = yTraj[0].flatten()
            self.trajectory[:,2] = xTraj[1].flatten()
            self.trajectory[:,3] = yTraj[1].flatten()
            self.trajectory[:,4] = xTraj[-1].flatten()


        elif trajSelection==2:  #Follow a set of points
            # This trajectory takes an array of random X,Y points and runs a trajectory between them
            self.trajectory = np.empty((0,5))

            prevTime = 0

            for ii in range(0,randomPoints.shape[0]-1):
                xTF = self.p2pMotionFinalTime(randomPoints[ii,0],randomPoints[ii+1,0])
                yTF = self.p2pMotionFinalTime(randomPoints[ii,1],randomPoints[ii+1,1])

                actualTF = max(xTF,yTF)

                if actualTF<(simulationTime/(randomPoints.shape[0]-1)):
                    actualTF=(simulationTime/(randomPoints.shape[0]-1))
                

                TF,xTraj = self.p2pMotionTrajectoryGenerator(randomPoints[ii,0],randomPoints[ii+1,0],actualTF)
                TF,yTraj = self.p2pMotionTrajectoryGenerator(randomPoints[ii,1],randomPoints[ii+1,1],actualTF)

                xTraj = np.array(xTraj)
                yTraj = np.array(yTraj)

                if xTraj.shape[0] > yTraj.shape[0]:
                    newY = np.ones(xTraj.shape)
                    newY[0:yTraj.shape[0]+1,:] = yTraj
                    for ii in range(0,yTraj.shape[1]):
                        newY[yTraj.shape[0]+1:,ii]=yTraj[-1,ii]
                    yTraj = newY
                elif xTraj.shape[0] < yTraj.shape[0]:
                    newX = np.ones(yTraj.shape)
                    newX[0:xTraj.shape[0]+1,:] = xTraj
                    for ii in range(0,xTraj.shape[1]):
                        newX[xTraj.shape[0]+1:,ii]=xTraj[-1,ii]
                    xTraj = newX

            
                # print(xTraj.shape)
                # print(yTraj.shape)

                trajectory = np.empty((xTraj[0].shape[0],5))
                trajectory[:,0] = xTraj[0].flatten()
                trajectory[:,1] = yTraj[0].flatten()
                trajectory[:,2] = xTraj[1].flatten()
                trajectory[:,3] = yTraj[1].flatten()
                trajectory[:,4] = xTraj[-1].flatten() + prevTime
                prevTime = trajectory[-1,4] + 1/self.sampFreq
                self.trajectory = np.vstack((self.trajectory,trajectory))






            # for ii in range(0,randomPoints.shape[0]-1):
            #     print(randomPoints[ii,:])
            #     print(randomPoints[ii+1,:])
            #     xTF = self.p2pMotionFinalTime(randomPoints[ii,0],randomPoints[ii+1,0])
            #     yTF = self.p2pMotionFinalTime(randomPoints[ii,1],randomPoints[ii+1,1])

            #     actualTF = max(xTF,yTF)

            #     if actualTF<float(simulationTime/randomPoints.shape[0]):
            #         actualTF = float(simulationTime/randomPoints.shape[0])
                
            #     print(actualTF)
            #     TF,xTraj = self.p2pMotionTrajectoryGenerator(randomPoints[ii,0],randomPoints[ii+1,0],actualTF)
            #     TF,yTraj = self.p2pMotionTrajectoryGenerator(randomPoints[ii,1],randomPoints[ii+1,1],actualTF)
            #     # print(xTraj)
            #     # print(yTraj)
            #     xTraj = np.array(xTraj).reshape((-1,5))
            #     yTraj = np.array(yTraj).reshape((-1,5))
                
                

            #     # if xTraj.shape[0]==2:
            #     #     xTraj = xTraj[0,:].reshape((1,5))
            #     # if yTraj.shape[0]==2:
            #     #     yTraj = yTraj[0,:].reshape((1,5))

            #     # print(yTraj)

            #     # if xTraj.shape[0] > yTraj.shape[0]:
            #     #     newY = np.ones(xTraj.shape)
            #     #     newY[0:yTraj.shape[0],:] = yTraj
            #     #     for ii in range(0,yTraj.shape[1]):
            #     #         newY[yTraj.shape[0]:,ii]=yTraj[-1,ii]
            #     #     yTraj = newY
            #     # elif xTraj.shape[0] < yTraj.shape[0]:
            #     #     newX = np.ones(yTraj.shape)
            #     #     newX[0:xTraj.shape[0],:] = xTraj
            #     #     for ii in range(0,xTraj.shape[1]):
            #     #         newX[xTraj.shape[0]:,ii]=xTraj[-1,ii]
            #     #     xTraj = newX
                
            #     # print(xTraj[-100:,0])
            #     # print(yTraj[-100:,0])
            #     tmpTraj = np.vstack((xTraj[:,0].flatten(),yTraj[:,0].flatten(),xTraj[:,1].flatten(),yTraj[:,1].flatten(),xTraj[:,-1].flatten())).T
            #     # print(tmpTraj.shape)
            #     self.trajectory = np.vstack((self.trajectory,tmpTraj))
            #     print(self.trajectory.shape)

            # # Now lets head back to the start
            # xTF = self.p2pMotionFinalTime(randomPoints[-1,0],randomPoints[0,0])
            # yTF = self.p2pMotionFinalTime(randomPoints[-1,1],randomPoints[0,1])

            # actualTF = max(xTF,yTF)

            # if actualTF<simulationTime/randomPoints.shape[0]:
            #     actualTF=simulationTime/randomPoints.shape[0]
            

            # TF,xTraj = self.p2pMotionTrajectoryGenerator(randomPoints[-1,0],randomPoints[0,0],actualTF)
            # TF,yTraj = self.p2pMotionTrajectoryGenerator(randomPoints[-1,1],randomPoints[0,1],actualTF)

            # xTraj = np.array(xTraj).reshape((-1,5))
            # yTraj = np.array(yTraj).reshape((-1,5))


            # # print(xTraj)
            # if xTraj.shape[0] > yTraj.shape[0]:
            #     newY = np.ones(xTraj.shape)
            #     newY[0:yTraj.shape[0],:] = yTraj
            #     for ii in range(0,yTraj.shape[1]):
            #         newY[yTraj.shape[0]:,ii]=yTraj[-1,ii]
            #     yTraj = newY
            # elif xTraj.shape[0] < yTraj.shape[0]:
            #     newX = np.ones(yTraj.shape)
            #     newX[0:xTraj.shape[0],:] = xTraj
            #     for ii in range(0,xTraj.shape[1]):
            #         newX[xTraj.shape[0]:,ii]=xTraj[-1,ii]
            #     xTraj = newX
            
            # tmpTraj = np.vstack((xTraj[:,0].flatten(),yTraj[:,0].flatten(),xTraj[:,1].flatten(),yTraj[:,1].flatten(),xTraj[:,-1].flatten())).T
            # # print(tmpTraj.shape)
            # self.trajectory = np.vstack((self.trajectory,tmpTraj))
            # print(self.trajectory.shape)
            # # print(self.trajectory.shape)






