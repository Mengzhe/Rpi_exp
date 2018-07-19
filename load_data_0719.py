# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:42:34 2018

@author: Mengzhe Huang
"""
import numpy as np
qq = np.array([[1, 0],
               [0, 0]])

# state 1: distance to center; state 2: curvature
dim_state = 2


def reward_func(state_current, action_current, state_succesor):
#    reward = -(state_succesor.transpose()@qq@state_succesor)
    reward = 1/(state_succesor.transpose()@qq@state_succesor + 1e1) 
    return reward

sars_tuple_for_all_trials = np.empty((0,6))
for i in range(50):
    filename = "./data/memory_%.3d.npy" % i
    memory = np.load(filename)
    memory = memory[:np.where(~memory.any(axis=1))[0][0],:]
    sars_tuple_single_trial = np.empty((0,6))
    
    for t in range(memory.shape[0]-1):
        state_current = memory[t, :dim_state]
        action_current = memory[t, dim_state]
        state_succesor = memory[t+1, :dim_state]
        reward = reward_func(state_current, action_current, state_succesor)
        tuple_current = np.hstack([state_current, action_current, reward, state_succesor])
        sars_tuple_single_trial = np.append(sars_tuple_single_trial, [tuple_current], axis=0)
    
    sars_tuple_for_all_trials =  np.append(sars_tuple_for_all_trials, sars_tuple_single_trial, axis=0)
    
np.save('sars_tuple_for_all_trials_01_0719', sars_tuple_for_all_trials)