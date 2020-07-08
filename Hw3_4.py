# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:37:29 2018

@author: kc
"""
import numpy
c = [-2, -1]
A = [[1, 1], [1, 2]]
b = [3, 5]

x0_bounds = (0, 2)
x1_bounds = (0, None)

from scipy.optimize import linprog
res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
               options={"disp": True})

print(res)
    
