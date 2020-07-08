import numpy as np
from scipy.optimize import minimize

def func(x, sign=1.0):
    return sign*(x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2 + 4*x[0]+5*x[1] + 6*x[2])    

def func_deriv(x, sign=1.0):
    dfdx0 = sign*(2*(x[0] + x[1] + 2))
    dfdx1 = sign*(2*x[0] + 6*x[1] + 5)
    dfdx2 = sign*(6)
    return np.array([dfdx0, dfdx1, dfdx2])

cons = ({'type':'eq',
         'fun' : lambda x: np.array([x[0] + 2*x[1] - 3]),
         'jac' : lambda x: np.array([1.0, 2.0, 0.0])},       
        {'type':'eq',
         'fun' : lambda x: np.array([4*x[0] + 5*x[2] - 6]),
         'jac' : lambda x: np.array([4.0, 0.0, 5.0])})

res = minimize(func, [1.0,1.0,1.0], args=(1.0), jac=func_deriv, constraints=cons,
               method='SLSQP', options={'disp': True})
print(res.x)