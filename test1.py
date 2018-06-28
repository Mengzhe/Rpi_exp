# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:25:17 2018

@author: Mengzhe Huang
"""

import numpy as np

# actions
l1 = 0
l2 = 50
l3 = 90
r1 = 0 
r2 = 50
r3 = 90
action_set = np.array([[l1, r1], 
                       [l1, r2],
                       [l1, r3],
                       [l2, r1],
                       [l2, r2],
                       [l2, r3],
                       [l3, r1],
                       [l3, r2],
                       [l3, r3]])
dim_action = action_set.shape[0]

chosen_action_number = np.random.randint(0,dim_action)
a = action_set[chosen_action_number]
pwm_l = a[0]
pwm_r = a[1]