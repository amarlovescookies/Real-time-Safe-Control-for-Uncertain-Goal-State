import numpy as np
import matplotlib.pyplot as plt
from math import factorial, atan2
from scipy.interpolate import PPoly

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver


import importlib

# import pos_constraints
# importlib.reload(pos_constraints)
# from pos_constraints import Ab_i1

def Ab_i1(i, n, d, dt_i, w_i, w_ip1):
    '''
    Ab_i1(i, n, d, dt_i, w_i, w_ip1) computes the linear equality constraint
    constants that require the ith polynomial to meet waypoints w_i and w_{i+1}
    at it's endpoints.
    Parameters:
        i - index of the polynomial.
        n - total number of polynomials.
        d - the number of terms in each polynomial.
        dt_i - Delta t_i, duration of the ith polynomial.
        w_i - waypoint at the start of the ith polynomial.
        w_ip1 - w_{i+1}, waypoint at the end of the ith polynomial.
    Outputs
        A_i1 - A matrix from linear equality constraint A_i1 v = b_i1
        b_i1 - b vector from linear equality constraint A_i1 v = b_i1
    '''
    # For each end point (start and finish), there are going to be position and velocity constraints
    # For each intermediary point, there are only going to be position constraints
    # For the car, there are 2 flat outputs, x and y. 
    numFlatOutputs = 2
    numOrderConstraints = int(d/2)
    numIntermediateConstraints = (n-1)*numFlatOutputs # This accounts for the position constraints for the intermediate points
    numConstraints = numFlatOutputs*numOrderConstraints*2 + numIntermediateConstraints # There are 2 constraints (Pos and Vel) and two end points (start and finish)
    conNum = 0
    # print(dt_i)

    A_i1 = np.zeros((numConstraints, numFlatOutputs*d*n))#,dtype='object')
    b_i1 = np.zeros((numConstraints, 1))

    
    # TODO: fill in values for A_i1 and b_i1

    # Here, we just set the sigma_i,0 to 1
    A_i1[conNum,2*d*i] = 1      # X_Start
    b_i1[conNum] = w_i[0]
    conNum+=1

    A_i1[conNum,2*d*i+1] = 1   # Y_Start
    b_i1[conNum] = w_i[1]
    conNum+=1

    if i==0:
        A_i1[conNum,2*d*i+2] = 1      # X_Velocity_Start
        b_i1[conNum] = w_i[2]
        conNum+=1
        A_i1[conNum,2*d*i+3] = 1     # Y_Velocity_Start
        b_i1[conNum] = w_i[3]
        conNum+=1
        if d>=6:        # Can use acceleration constraints
            A_i1[conNum,2*d*i+4] = 2      # X_Acceleration_Start
            b_i1[conNum] = w_i[4]
            conNum+=1
            A_i1[conNum,2*d*i+5] = 2     # Y_Acceleration_Start
            b_i1[conNum] = w_i[5]
            conNum+=1


    
    for ii in range(0,d):
        A_i1[conNum,2*d*i+ii*2] = dt_i**ii      # X_End
        conNum+=1
        A_i1[conNum,2*d*i+1+ii*2] = dt_i**ii    # Y_End
        conNum-=1


    b_i1[conNum] = w_ip1[0]
    conNum+=1
    b_i1[conNum] = w_ip1[1]
    conNum+=1

    
    if i==n-1:
        for ii in range(0,d):
            A_i1[conNum,2*d*i+ii*2] = ii*dt_i**np.clip(ii-1,0,999)      # X_Velocity_End
            conNum+=1
            A_i1[conNum,2*d*i+1+ii*2] = ii*dt_i**np.clip(ii-1,0,999)    # Y_Velocity_End

            if d>=6:        # Can use acceleration constraints
                conNum+=1
                # print(2*d*i+2+ii*2)
                A_i1[conNum,2*d*i+ii*2] = ii*np.clip(ii-1,0,999)*dt_i**np.clip(ii-2,0,999)      # X_Acceleration_End
                # b_i1[conNum] = 0
                conNum+=1
                A_i1[conNum,2*d*i+1+ii*2] = ii*np.clip(ii-1,0,999)*dt_i**np.clip(ii-2,0,999)     # Y_Acceleration_End
                # b_i1[conNum] = 0
                conNum-=2
                



            conNum-=1


        b_i1[conNum] = w_ip1[2]
        conNum+=1
        b_i1[conNum] = w_ip1[3]
        conNum+=1
        b_i1[conNum] = w_ip1[4]
        conNum+=1
        b_i1[conNum] = w_ip1[5]
        conNum+=1
    tmp=[]
    for ii in range(0,A_i1.shape[0]):
        if not A_i1[ii,:].any():
            break

    A_i1 = A_i1[0:ii,:]
    b_i1 = b_i1[0:ii,:]

    # print(A_i1)
    # print(b_i1)
    return A_i1, b_i1


def add_pos_constraints(prog, sigma, n, d, w, dt):
    # Add A_i1 constraints here
    for i in range(n):
        Aeq_i, beq_i = Ab_i1(i, n, d, dt[i], w[i], w[i + 1])
        # print(f'A{i}: {Aeq_i}')
        # print(f'dt{i}: {dt[i]}')
        # print(f'w{i}: {w[i]}')
        prog.AddLinearEqualityConstraint(Aeq_i, beq_i, sigma.flatten())
        

def add_continuity_constraints(prog, sigma, n, d, dt):
    # TDOO: Add A_i2 constraints here
    # Hint: Use AddLinearEqualityConstraint(expr, value)
    from math import factorial



    # print(sigma.shape)
    for kk in range(1,5):
        for ii in range(0,n-1):
            eq0 = -factorial(kk)*sigma[ii+1,kk]
            # eq1 = -factorial(kk)*sigma[ii+1,kk,1]
            for jj in range(kk,d):
                eq0 += (factorial(jj)/factorial(jj-kk))*sigma[ii,jj]*(dt[ii]**(jj-kk))
                # eq1 += (factorial(jj)/factorial(jj-kk))*sigma[ii,jj,1]*(dt[ii]**(jj-kk))
 
            # print(eq0)
            prog.AddLinearEqualityConstraint(eq0,[0,0])
            # prog.AddLinearEqualityConstraint(eq1,0)


    pass
  
def add_minsnap_cost(prog, sigma, n, d, dt):
    # TODO: Add cost function here
    # Use AddQuadraticCost to add a quadratic cost expression
    from math import factorial

    derNum = 2

    E0 = 0
    E1 = 0
    for ii in range(0,n):
        for jj in range(derNum,d):
            for kk in range(derNum,d):
                cTerm = ((factorial(jj)/factorial(jj-derNum))*(factorial(kk)/factorial(kk-derNum)))
                # sigTerm0 = sigma[ii,jj,0]*sigma[ii,kk,0]
                # sigTerm1 = sigma[ii,jj,1]*sigma[ii,kk,1]
                sigTerm = sigma[ii,jj]@sigma[ii,kk]
                tTermT = (dt[ii]**(jj+kk-(derNum*2-1)))/(jj+kk-(derNum*2-1))
                E0+= cTerm*sigTerm*tTermT
                
              
    E = E0+E1



    prog.AddQuadraticCost(E)

    pass

def minsnap(n, d, w, dt):
    n_dim = 2
    dim_names = ['x', 'y']

    prog = mp.MathematicalProgram()
    # sigma is a (n, n_dim, d) matrix of decision variables
    sigma = np.zeros((n, d, n_dim), dtype="object")
    dt2 = np.zeros((n),dtype="object")
    for i in range(n):
        # dt2[i] = prog.NewContinuousVariables(1, "dt_" + str(i))
        for j in range(d):
            sigma[i][j] = prog.NewContinuousVariables(n_dim, "sigma_" + str(i) + ',' +str(j)) 
        
    # print(dt2[0][0])
    add_pos_constraints(prog, sigma, n, d, w, dt)#2[0])

    # If we only have one segment, there is no need for continuity constraints
    if n!=1:
        add_continuity_constraints(prog, sigma, n, d, dt)
        
    add_minsnap_cost(prog, sigma, n, d, dt)#2[0])  

    # print(pydrake.quadratic_costs(prog))
    solver = OsqpSolver()
    result = solver.Solve(prog)
    # print(result.get_solution_result())
    v = result.GetSolution()
    # print(v)
    # Reconstruct the trajectory from the polynomial coefficients
    coeffs_y = v[::2]
    coeffs_z = v[1::2]
    y = np.reshape(coeffs_y, (d, n), order='F')
    z = np.reshape(coeffs_z, (d, n), order='F')
    coeff_matrix = np.stack((np.flip(y, 0), np.flip(z, 0)), axis=-1)  
    t0 = 0
    t = np.hstack((t0, np.cumsum(dt)))
    minsnap_trajectory = PPoly(coeff_matrix, t, extrapolate=False)

    return minsnap_trajectory