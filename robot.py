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
import importlib


import spline
importlib.reload(spline)
from spline import minsnap



class Robot(object):
    def __init__(self,arenaSize,targetMotConstraints,initialConditions,finalConditions,intermediatePoints=[],order=4,tolerance=0.5, predictionSteps = 10):
        self.setMachineParameters(targetMotConstraints)
        self.n = 1+len(intermediatePoints)
        self.d = order
        self.currentState = np.hstack((initialConditions,[0,0]))
        self.targetPos = finalConditions[0:2]#finalConditions[0:2]

        self.targetArray = np.empty((predictionSteps,2))
        self.predictionSteps = predictionSteps
        self.targetVelocity = np.zeros((2,))
        self.targetAcceleration = np.zeros((2,))
        self.targetJerk = np.zeros((2,))

        self.wStar = initialConditions[0:2]
        self.prev_wStar = self.wStar
        # Need to initialize the trajectory array
        self.trajectory = np.zeros((2,8))
        self.trajectory[0,0:6] = initialConditions
        self.trajectory[1,0:6] = initialConditions
        self.finalTrajectory = np.empty((0,8))#initialConditions[0:4]
        self.prevTraj = np.array([initialConditions,initialConditions])
        self.tolerance = tolerance
        self.trajCounter = 1
        self.trajectoryGeneratedFlag = 0
        # For now, only using two waypoints, so a single segment
        if intermediatePoints==[]:
            self.w = np.array([initialConditions,finalConditions])
        else:
            self.w = np.array([initialConditions,intermediatePoints,finalConditions])
        
        # print(f'Starting waypoint size: {self.w.shape}')
        


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

    def addWaypoint(self,waypoint,position):
        if position==0:
            self.w = np.vstack((waypoint,self.w))
        else:
            self.w = np.vstack((self.w[0:position,:],waypoint,self.w[position:,:]))

        # Also need to update n
        self.n = self.w.shape[0]-1

        return 0

    def add2TargetArray(self,targetPos):
        self.targetArray = np.vstack((self.targetArray[1:self.predictionSteps,:],targetPos))
        vel,acc,jerk = self.calcJAV()
        self.targetVelocity = np.mean(vel,axis=0)
        self.targetAcceleration = np.mean(acc,axis=0)
        self.targetJerk = np.mean(jerk,axis=0)

        # print(self.targetArray.shape)


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

    def addProtrusionPoint(self):
        '''
        This function will add a waypoint to shift the trajectory so that we do not enter the uncertainty zone
        '''

        midpoint = [(self.currentState[0]+self.targetPos[0])/2,(self.currentState[1]+self.targetPos[1])/2]
        
        wStarDist = np.sqrt((self.wStar[0]-self.targetPos[0])**2 + (self.wStar[1]-self.targetPos[1])**2)
        # print(wStarDist)

        # Need to find where wStar is relative to current position
        targetAngle = np.arctan2((self.targetPos[1]-self.currentState[1]),(self.targetPos[0]-self.currentState[0]))
        wStarAngle = np.arctan2((self.wStar[1]-self.currentState[1]),(self.wStar[0]-self.currentState[0]))

        if wStarAngle<targetAngle:
            # Need to put the point at a -90 degree angle 
            protAngle = wStarAngle-np.pi/2.
        else:
            protAngle = wStarAngle+np.pi/2.    

        protPoint = np.array([np.cos(protAngle),np.sin(protAngle),0.,0.,0.,0.])*wStarDist
        protPoint[0:2] = protPoint[0:2] + midpoint


        self.addWaypoint(protPoint,1)
        return 0
        # print('hello')
        
    def updateTargetPos(self, mean, wStar):
        '''
        This function will update the target position and the wStar
        '''
        self.targetPos = mean
        self.wStar = wStar
        self.w[-1,0:2] = wStar
        self.add2TargetArray(mean)

    def updateCurrentState(self):
        self.currentState[0:8] = self.trajectory[self.trajCounter,0:8]
        # Need to restrict Velocity, Acceleration, Jerk
        self.currentState[2] = np.clip(self.trajectory[self.trajCounter,2],-self.Vmax,self.Vmax)
        self.currentState[3] = np.clip(self.trajectory[self.trajCounter,3],-self.Vmax,self.Vmax)
        self.currentState[4] = np.clip(self.trajectory[self.trajCounter,4],-self.Amax,self.Amax)
        self.currentState[5] = np.clip(self.trajectory[self.trajCounter,5],-self.Amax,self.Amax)
        self.currentState[6] = np.clip(self.trajectory[self.trajCounter,4],-self.Jmax,self.Jmax)
        self.currentState[7] = np.clip(self.trajectory[self.trajCounter,5],-self.Jmax,self.Jmax)
        self.w[0,:] = self.currentState[0:6]
        self.finalTrajectory = np.vstack((self.finalTrajectory,self.currentState[0:8]))
        if self.trajectoryGeneratedFlag:
            self.trajCounter+=1
        # print(w.shape)

    def getCurrentState(self):
        return (self.currentState[0],self.currentState[1],self.currentState[2],self.currentState[3])

    def updateUncertainty(self,uncertainty=np.zeros((11,))):
        '''
        This adds the uncertainty parameters:
        Index 0 = Mean X
        Index 1 = Mean Y
        Index 2 = W1 X
        Index 3 = W1 Y
        Index 4 = W2 X
        Index 5 = W2 Y
        Index 6 = W3 X
        Index 7 = W3 Y
        Index 8 = W4 X
        Index 9 = W4 Y
        Index 10 = Theta
        '''
        self.uncertainty = uncertainty

    def calcJAV(self):
        if self.targetArray.shape[0] != self.predictionSteps:
            return -1
        else:
            currentInd = np.arange(0,self.predictionSteps-1)
            nextInd = np.arange(1,self.predictionSteps)
            
            velocity = self.targetArray[nextInd,:]-self.targetArray[currentInd,:]
            acceleration = velocity[nextInd[0:(self.predictionSteps-2)],:]-velocity[currentInd[0:(self.predictionSteps-2)],:]
            jerk = acceleration[nextInd[0:(self.predictionSteps-3)],:]-acceleration[currentInd[0:(self.predictionSteps-3)],:]
        
        
        # Now we can return the calculated Velocity, Acceleration, and Jerk
        return velocity,acceleration,jerk

    def kinematicPositionEstimate(self):
        futurePos = np.zeros((2,))
        futureVel = np.zeros((2,))
        futureAcc = np.zeros((2,))
        if self.targetArray.shape[0]==self.predictionSteps:
            vel,acc,jerk = self.calcJAV()

            vNew = np.mean(vel,axis=0)
            aNew = np.mean(acc,axis=0)
            J = np.mean(jerk,axis=0)
            # print(vNew)
            # print(aNew)
            # print(J)

            t = ((1/self.sampFreq)*self.predictionSteps)
            # print(t)
            # With the current vel, acc, and jerk, we can model the target out to some time in the future
            futurePos = self.wStar + vNew*t + 0.5*aNew*t**2 + (1./3.)*J*t**3
            futureVel = vNew + aNew*t + 0.5*J*t**2
            futureAcc = aNew + J*t

        # print(self.targetArray)

        return futurePos,futureVel,futureAcc

    def updateFinalWaypointWithEstimate(self):
        futurePos, futureVel, futureAcc = self.kinematicPositionEstimate()
        self.w[-1,:] = np.hstack((futurePos,futureVel,futureAcc))
        # print(self.w[-1,:])

    def resetW(self):
        self.w = np.array([[self.currentState[0:6]],[np.hstack((self.wStar,np.zeros((4,))))]]).reshape((2,6))
        # print(self.w)
        self.n = 1

    def createTrajectory(self,scaleFlag=True,sophisticatedTimeScalingFlag=True,mpcFlag=False):
        '''
        This function will return the trajectory from the current position to the goal position
        '''
        # if np.sqrt((self.wStar[0]-self.prev_wStar[0])**2 + (self.wStar[1]-self.prev_wStar[1])**2)<self.tolerance:
        #     return self.trajectory
        self.trajectoryGeneratedFlag = 1
        self.prevTraj = self.trajectory
        self.trajCounter = 1
        # if self.currentState[0]==self.targetPos[0] and self.currentState[1]==self.targetPos[1]:
        #     return 0
        
        # Need to reset the waypoint vector:
        self.resetW()





        if mpcFlag==True:
            self.updateFinalWaypointWithEstimate()

        target2WSTARAngle = np.arctan2(self.targetPos[1]-self.wStar[1],self.targetPos[0]-self.wStar[0])
        target2robotAngle = np.arctan2(self.targetPos[1]-self.currentState[1],self.targetPos[0]-self.currentState[0])
        
        if abs(target2WSTARAngle-target2robotAngle)>np.pi/3:
            self.addProtrusionPoint()


        
        
        
        dt = np.zeros(self.n) # Will change based on max velocity of calculated trajectory
        for ii in range(0,self.n):
            if sophisticatedTimeScalingFlag==1:
                distance2Target = np.sqrt((self.w[ii,0] - self.w[ii+1,0])**2+(self.w[ii,1] - self.w[ii+1,1])**2)
                dt[ii] = self.p2pMotionFinalTime(0,distance2Target)
            else:
                dt[ii]=1

        # print(f'dist2Target: {distance2Target}')

        minsnap_trajectory = minsnap(self.n, self.d, self.w, dt)
        
        


        # Want to go through each segment and ensure the max velocity has not been violated
        if scaleFlag:
            t0=0
            tf=1
            n_points = 100
            timeScaler = []
            for ii in range(self.n):
                t = np.linspace(t0, tf, n_points)
                timeScaler.append(np.max(minsnap_trajectory(t,1)[:])/self.Vmax)
                # If we use the more sophisticated scaler
                # distance2Target = np.sqrt((self.w[ii,0] - self.w[ii+1,0])**2+(self.w[ii,1] - self.w[ii+1,1])**2)
                # timeScaler.append(self.p2pMotionFinalTime(0,distance2Target))




                dt[ii]= dt[ii]*timeScaler[ii]
                t0+=tf
                tf+=tf
            
            # With dt scaled, can calculate the actual trajectory
            
            minsnap_trajectory = minsnap(self.n, self.d, self.w, dt)

        t0=0
        tf = sum(dt)
        # print(tf)
        # print(dt)
        n_points = self.sampFreq*tf
        t = np.linspace(t0, tf, n_points)

        
        pos = minsnap_trajectory(t,0)[:]
        vel = minsnap_trajectory(t,1)[:]
        acc = minsnap_trajectory(t,2)[:]
        jerk = minsnap_trajectory(t,3)[:]
        
        # print(pos.shape)
        trajectory = np.hstack((pos,vel,acc,jerk,t.reshape((-1,1))))
        self.trajectory = np.hstack((pos,vel,acc,jerk,t.reshape((-1,1))))

        # Animation function only needs position, velocity, and time, and relies on values from object,
        # So in order to keep basic functionality identical, I'm setting the object trajectory for use with
        # animation, and outputing all the derivatives of position from this function
        return trajectory
    
    def calc_wStar(self, w1, w2, w3, w4):
        """
        wStar()
        Calculates the target location based on the direction of minimum uncertainty

        Inputs:
        w1 - 2x1 numpy array - direction of uncertainty
        w2 - 2x1 numpy array - direction of uncertainty
        mu - 2x1 numpy array - mean location of target

        Returns: 
        wStar - 2x1 numpy array
        """
 
        # Find the direction of least uncertainty
        w1Mag = np.linalg.norm(w1)
        w2Mag = np.linalg.norm(w3)

        if w1Mag < w2Mag:
            pt1_w = w1
            pt2_w = w2
        else:
            pt1_w = w3
            pt2_w = w4


        # Calculate distance from w* to robot
        pt_robot = self.currentState[0:2]
        pt1_dist = np.linalg.norm(pt_robot - pt1_w)
        pt2_dist = np.linalg.norm(pt_robot - pt2_w)

        # Finds the shortest distance
        if pt1_dist < pt2_dist:
            wStar = pt1_w
        else:
            wStar = pt2_w
            
        return wStar 

    def omega_star(self, prev_wStar, alpha, particles, weights, mean, w1, w2, w3, w4):
        """ 
        Calculates the target location based on the direction of minimum uncertainty
        Inputs:
        prev_wStar - 2x1 numpy array of prev wStar
        alpha - threshold for determining if wStar needs update
        particles - Nx2 array of coordinates of particles
        weights - Nx1 array of weights of particles
        mean - mean of probability ditribution of particles
        w1 - 2x1 numpy array - direction of uncertainty
        w2 - 2x1 numpy array - direction of uncertainty
        w3 - 2x1 numpy array - direction of uncertainty
        w4 - 2x1 numpy array - direction of uncertainty
        Returns: 
        wStar - 2x1 numpy array
        """
        robot = self.currentState[0:2]
        # only for initialization

        particle_max_wt = 'particle' 
        min_ax = 'min'
        max_ax = 'max'

        # Initialize min and max
        min_ax1 = np.zeros((1,2))
        min_ax2 = np.zeros((1,2))
        max_ax1 = np.zeros((1,2))
        max_ax2 = np.zeros((1,2))
        wStar = np.zeros((1,2))



        # We can choose which point we want as wStar to analyze dist to true target and probability of collision
        # omega = 'mean'
        # omega  = particle_max_wt
        omega = 'min'
        # omega = max_ax

        if np.linalg.norm(mean - w1) < np.linalg.norm(mean - w3): # checking which points lie along min or max axis
            min_ax1 = w1
            min_ax2 = w2
            max_ax1 = w3
            max_ax2 = w4
        else:
            min_ax1 = w3
            min_ax2 = w4
            max_ax1 = w1
            max_ax2 = w2
        
        if omega=='mean':
            wStar = mean

        elif omega=='particle':
            particle_max_wt = particles[np.argmax(weights)] # Partcile with maximum weight
            wStar = particle_max_wt

        elif omega=='min':
            if np.linalg.norm(robot - min_ax1) < np.linalg.norm(robot - min_ax2): # check which is closer to robot and assign that to be min_ax
                min_ax = min_ax1
            else:
                min_ax = min_ax2
            wStar = min_ax

        elif omega=='max':
            if np.linalg.norm(robot - max_ax1) < np.linalg.norm(robot - max_ax2): # check which is closer to robot and assign that to be max_ax
                max_ax = max_ax1
            else:
                max_ax = max_ax2
            wStar = max_ax

        elif omega=='follow':
            targetTravelAngle = np.arctan2(self.targetArray[-1,1]-self.targetArray[0,1],self.targetArray[-1,0]-self.targetArray[0,0])*180/np.pi
            minAx1Angle = np.arctan2(min_ax1[1]-mean[1],min_ax1[0]-mean[0])*180/np.pi
            minAx2Angle = np.arctan2(min_ax2[1]-mean[1],min_ax2[0]-mean[0])*180/np.pi

            minAngle = 0.05
            
            if abs(targetTravelAngle-minAx1Angle) > abs(targetTravelAngle-minAx2Angle) and abs(targetTravelAngle-minAx1Angle) - abs(targetTravelAngle-minAx2Angle) > minAngle:
                wStar = min_ax1
            elif abs(targetTravelAngle-minAx1Angle) < abs(targetTravelAngle-minAx2Angle) and abs(targetTravelAngle-minAx1Angle) - abs(targetTravelAngle-minAx2Angle) > minAngle:
                wStar = min_ax2
            elif abs(targetTravelAngle-minAx1Angle) - abs(targetTravelAngle-minAx2Angle) <= minAngle:
                wStar = min_ax1
                


        else:
            wStar = mean

        if np.linalg.norm(wStar - prev_wStar) < alpha: # if new wStar is within alpha distance of previous wStar, do not update
            return prev_wStar
        else:
            return wStar  
                    