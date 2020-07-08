# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:20:14 2018

@author: kc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:56:12 2018

@author: kc
"""

import numpy as np
from scipy.optimize import minimize

def func(x, sign=1.0):
    return sign*(4*x[0]**2+4*x[1]**2+3*x[2]**2+3*x[0]*x[1]+4*x[0]*x[2]+5*x[1]*x[2]-4*x[0]-6*x[1]-9*x[2])    

def func_deriv(x, sign=1.0):
    dfdx0 = sign*(8*x[0]+3*x[1]+4*x[2]-4)
    dfdx1 = sign*(8*x[1]+3*x[0]+5*x[2]-6)
    dfdx2 = sign*(6*x[2]+4*x[0]+5*x[1]-9)
    return np.array([ dfdx0, dfdx1, dfdx2])

cons = ({'type':'ineq',
         'fun' : lambda x: np.array([-x[0]-2*x[1]-x[2]+4]),
         'jac' : lambda x: np.array([-1.0, -2.0, -1.0])},
        
        {'type':'ineq',
         'fun' : lambda x: np.array([-2*x[0]-x[1]+x[2]+5]),
         'jac' : lambda x: np.array([-2.0, -1.0, 1.0])})

res = minimize(func, [1.0,1.0,1.0], args=(1.0,), jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
print(res.x)