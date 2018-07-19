#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:25:59 2018

@author: mengzhehuang
"""

import numpy as np
memory = np.load('sars_tuple_for_all_trials_0719.npy')
actions_all = memory[:,2]
#action_counts = np.histogram(actions_all, bins=[0, 1, 2, 3, 4])
unique, counts = np.unique(actions_all, return_counts=True)


## state 1: distance to center; state 2: curvature
#dim_state = 2
#dim_action = 5
#
#def feature_pre(s):
#    s_transpose = s.transpose()
#    # feature_pre is the kernel function of the state
##    feature_sub = np.hstack((1, s_transpose, s_transpose**2, \
##                             [s_transpose[:,0]*s_transpose[:,1]])).transpose()
#    feature_sub = np.hstack((np.eye(1), s_transpose, s_transpose**2,[s_transpose[:,0]*s_transpose[:,1]])).transpose()
##    feature_sub = feature_sub[:, np.newaxis]
#    return feature_sub
#
#def feature_x(s, chosen_action_number):    
#    # feature_pre is the kernel function of the state 
#    feature_sub = feature_pre(s)
#    u = np.zeros((dim_action, 1))
#    chosen_action_number = np.int(chosen_action_number)
#    u[chosen_action_number, 0] = 1
#    # feature is the mapping function of state-action pair
#    feature = np.kron(u, feature_sub)
#    return feature
#
#def action_from_policy_pi(s, w):
#    # return the action if policy pi_w is used
#    # based on all possible Q values
#    feature_stack = np.zeros((dim_action, dim_feature))
#    
#    # we can use np.kron to simplify the code
#    # feature_stack = np.kron(np.eye(dim_action), s)
##    for i in range(dim_action):
##        feature_stack[i, :] = feature_x(s, i).transpose().squeeze()
##    feature_stack_test = np.kron(np.eye(dim_action), feature_pre(s).transpose())
##    norm_test = np.linalg.norm(feature_stack_test - feature_stack)
#    
#    feature_stack = np.kron(np.eye(dim_action), feature_pre(s).transpose())
#    q_values_from_all_actions = feature_stack @ w
#    action_number = np.argmax(q_values_from_all_actions)
#    return action_number
#
## initialization
#s = memory[0, :dim_state] 
#s = s[np.newaxis, :].transpose()
#a = memory[0, dim_state] 
#s_ = memory[0, -dim_state:] 
##feature_s = feature_pre(s)
#feature_init = feature_x(s, a)
#dim_feature = feature_init.shape[0]
## initialize parameters to [-1, 1]
#w_old = 2*np.random.rand(dim_feature, 1) - 1
#w_new = np.zeros((dim_feature, 1))
#
#
## computation part
#gamma = 0.9
#iter_max = 20
#w_hist = np.zeros((iter_max+1, dim_feature))
#w_hist[0,:] = w_old.squeeze()
#for iter in range(iter_max):
#    phi = np.zeros((dim_feature, dim_feature))
#    b = np.zeros((dim_feature, 1))
#    for t in range(memory.shape[0]):
#        state_current = memory[t, 0:dim_state]
#        state_current = state_current[np.newaxis, :].transpose()
#        action_current = np.int(memory[t, dim_state])
#        
#        reward = memory[t, dim_state+1]
#        
#        state_succesor = memory[t, -dim_state:]
#        state_succesor = state_succesor[np.newaxis, :].transpose()
#        action_succesor = np.int(action_from_policy_pi(state_succesor, w_old))
#        
#        x_current = feature_x(state_current, action_current)
#        x_successor = feature_x(state_succesor, action_succesor)
#        
##        temp = x_current @ (x_current - gamma*x_successor).transpose()
#        phi = phi + x_current @ (x_current - gamma*x_successor).transpose()
#        b = b + x_current*reward
#        
#    w_new = np.linalg.inv(phi) @ b
#    w_hist[iter+1,:] = w_new.squeeze()
#    w_old = w_new
## computation part ends
#np.save('learned_weights_0718', w_new)