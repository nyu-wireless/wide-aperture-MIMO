# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 07:48:49 2021

@author: Yaqi, Mingsheng
"""
import numpy as np

class PhyConst(object):
    """
    Physical constants
    """

    # light_speed = 2.99792458e8
    kT =  -174
    light_speed_remcom = 2.996955055081703e8


class LinkState(object):
    """
    Static class with link states
    """
    no_link = 0
    los_link = 1
    nlos_link = 2
    name = ['NoLink', 'LOS', 'NLOS']
    nlink_state = 3

    
class DataConfig(object):
    """
    Meta data on ray tracing data
    """
    freq = 28e9
    npaths_max = 25
    tx_pow_dbm = 23.0
    rx_noise_config = 3.0 # db

    total_received_power = 'total_received_power'
    mean_arrival_time = 'mean_arrival_time'
    spread_delay = 'spread_delay'
    paths_number = 'paths_number'
    path_rcvpow = '_srcvdpower' # dBm
    path_phase = '_phase'
    path_dly = '_arrival_time'
    path_aoa_theta = '_arrival_angle1' # inclined angle
    path_aoa_phi = '_arrival_angle2'
    path_aod_theta = '_departure_angle1' # inclined angle 
    path_aod_phi = '_departure_angle2'
    path_interact = '_interactions_list'
    path_num_interact = '_n_interactions'
    tx_coord = 'Tx_coordinates'
    rx_coord = 'Rx_coordinates'
    
    


def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
           [ 0, np.cos(theta),-np.sin(theta)],
           [ 0, np.sin(theta), np.cos(theta)]], dtype=object)

def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0                 , 1, 0           ],
                   [-np.sin(theta)   , 0, np.cos(theta)]], dtype=object)
  
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]], dtype=object)

def Qz(s):
    return np.array([[ 1, 0, 0],
                   [ 0, 1, 0],
                   [0, 0, s]], dtype=object)











        