# -*- coding: utf-8 -*-
"""
Chan object for buliding a channel
@author: Yaqi, Mingsheng

Example to run code for 140GHz with add_foliage_add_diffraction:
    python channel.py --center_freq 140e9 --case_name add_foliage_add_diffraction --n_freq 10
Example to run code for 140GHz with no_foliage_no_diffraction:
    python channel.py --center_freq 140e9 --case_name no_foliage_no_diffraction --n_freq 10
Example to run code for 28GHz with add_foliage_add_diffraction:
    python channel.py --center_freq 28e9 --case_name add_foliage_add_diffraction --n_freq 10
Example to run code for 28GHz with no_foliage_no_diffraction:
    python channel.py --center_freq 28e9 --case_name no_foliage_no_diffraction --n_freq 10
"""

from constants import LinkState, DataConfig, PhyConst, Rx, Ry, Rz, Qz

import numpy as np
import pandas as pd
import scipy.optimize
import random
import argparse

class MPChan(object):
    
    """
    Class for a multi-path channel from ray tracing
    """
    nangle = 4
    aoa_phi_ind = 0     #[-pi, pi]
    aoa_theta_ind = 1   #[-pi/2, pi/2]
    aod_phi_ind = 2
    aod_theta_ind = 3
    ang_name = ['AoA_Phi', 'AoA_theta', 'AoD_phi', 'AoD_theta']
    
    def __init__(self, data_frame, fc, is_outage = False):
        """
        MPChan Constructor from data_frame
        
        -----------
        npath : number of path of this MPChan
        XX_coord : np.array like [x, y, z]
        dly : second
        pl : dBm
        phase : degree
        angles : radian
        interact : string like "Tx-R-R-Rx"
        num_interaction : int number of interaction LOS = 0
        link_state : LinkState values
        
        -----------
        Function:
            compute_RM_error : Compute RM error
            compute_PWA_error : Compute PWA error
            compute_const_error : Compute Constant Model error

        """
        
        # Parameters for each path
        if is_outage:
            self.link_state = LinkState.no_link
        else:
            self.rnd_freq = fc
            self.npath = int(data_frame.at[0, DataConfig.paths_number])
            self.tx_coord = data_frame.at[0, DataConfig.tx_coord][1:-1]
            self.tx_coord = np.fromstring(self.tx_coord, sep=' ',dtype = float)
            self.rx_coord = data_frame.at[0, DataConfig.rx_coord][1:-1]
            self.rx_coord = np.fromstring(self.rx_coord, sep=' ',dtype = float)
            self.pl = np.zeros((self.npath)) # db
            self.dly = np.zeros((self.npath)) # second
            self.phase = np.zeros((self.npath))
            self.ang = np.zeros((self.npath, MPChan.nangle)) #rad
            self.interact = [] # interaction list wiht string
            self.num_interact = np.zeros((self.npath))
            
            for ipath in range(self.npath):
                path_id = ipath + 1 # path id starts from 1
                self.pl[ipath] = DataConfig.tx_pow_dbm - \
                    10*np.log10(1000*data_frame.at[0, str(path_id)+DataConfig.path_rcvpow]) # dBm
                self.dly[ipath] = data_frame.at[0, 
                        str(path_id)+DataConfig.path_dly]
                self.phase[ipath] = data_frame.at[0, 
                                  str(path_id)+DataConfig.path_phase]
                self.ang[ipath, 0] = np.deg2rad(\
                        data_frame.at[0, str(path_id)+DataConfig.path_aoa_phi])
                self.ang[ipath, 1] = np.deg2rad(90 
                        -data_frame.at[0, str(path_id)+DataConfig.path_aoa_theta])
                self.ang[ipath, 2] = np.deg2rad(
                        data_frame.at[0, str(path_id)+DataConfig.path_aod_phi])
                self.ang[ipath, 3] = np.deg2rad(90 
                        -data_frame.at[0, str(path_id)+DataConfig.path_aod_theta])
                self.interact.append(\
                         data_frame.at[0, str(path_id)+DataConfig.path_interact])
                self.num_interact[ipath] = data_frame.at[0, 
                         str(path_id)+DataConfig.path_num_interact]
            
            if 0 in self.num_interact: # LOS path has 0 interactions
                self.link_state = LinkState.los_link
            else:
                self.link_state = LinkState.nlos_link
    
    
    def compute_RM_error(self, ref_chan, valdn_chan, center_freq):
        '''
        Compute Reflection Model error

        Parameters
        ----------
        ref_chan : MPChan
            Reference MPChan
        valdn_chan : MPChan
            Validation MPChan

        Returns
        -------
        rel_err

        '''
        
        # match paths and compute gamma values
        self.compute_gamma(ref_chan)
        phase_ls = []
        gain_ls = []
        dly_ls = []
        s_value = [-1, 1]
        
        for i_path in range(ref_chan.npath):
            
            aoa_az = ref_chan.ang[i_path, 0]
            aoa_el = ref_chan.ang[i_path, 1]
            aod_az = ref_chan.ang[i_path, 2]
            aod_el = ref_chan.ang[i_path, 3]
            
            # see if we have find matched path in gamma channel
            if i_path in self.path_map:
            
                # choose gamma and s by reflecion times
                interact = ref_chan.interact[i_path].split('-')
                s_id = interact.count('R')%2
                
                path_id = self.path_map.index(i_path)
                gamma = self.gamma[path_id, s_id]
                
                # compute distance by gamma with valdn_chan_tx_rx_coord
                ref_path_distance = ref_chan.dly[i_path] * PhyConst.light_speed_remcom
                item_1 = np.array([ref_path_distance, 0, 0])
                item_2 = Ry(aoa_el).dot(Rz(-aoa_az)).dot(ref_chan.rx_coord-valdn_chan.rx_coord)
                item_3 = Qz(s_value[s_id]).dot(Rx(gamma)).dot(Ry(aod_el)).dot(Rz(-aod_az)).dot(ref_chan.tx_coord-valdn_chan.tx_coord)
                eq_right_vector = item_1 + item_2 + item_3
                
                est_dist = np.sqrt(np.sum(eq_right_vector**2))    
                
            else:
                # no path matched so use PWA
                # direction of arrival
                u_r = np.array([np.cos(aoa_az)*np.cos(aoa_el), 
                              np.sin(aoa_az)*np.cos(aoa_el),
                              np.sin(aoa_el)]) 
                
                # direction of departure
                u_t = np.array([np.cos(aod_az)*np.cos(aod_el), 
                              np.sin(aod_az)*np.cos(aod_el),
                              np.sin(aod_el)]) 
                
                # pwa approximated distance
                est_dist = PhyConst.light_speed_remcom * ref_chan.dly[i_path] +\
                        u_r.dot(ref_chan.rx_coord - valdn_chan.rx_coord) +\
                        u_t.dot(ref_chan.tx_coord - valdn_chan.tx_coord) 
            
            
            # compute arbitrary phase
            phase_ = 2*np.pi -(2*np.pi*center_freq*ref_chan.dly[i_path])%(2*np.pi)
            arbitrary_phase = np.deg2rad(ref_chan.phase[i_path]) - phase_
            
            phase_from_est_dist = 2*np.pi - (2*np.pi*est_dist*center_freq/PhyConst.light_speed_remcom) % (2*np.pi)
            phase_from_est_dist = phase_from_est_dist + arbitrary_phase
            dly_from_est_dist = est_dist/PhyConst.light_speed_remcom
            
            phase_ls.append(phase_from_est_dist)
            gain_ls.append(-ref_chan.pl[i_path])
            dly_ls.append(dly_from_est_dist)
            
        Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, ref_chan.rnd_freq, center_freq)
        Htrue, Etrue = comp_cmplx_gain(-valdn_chan.pl, np.deg2rad(valdn_chan.phase), valdn_chan.dly, valdn_chan.rnd_freq, center_freq)
        rel_err = np.abs(Hest-Htrue)**2/Eest
        print(f'RM Rel_err = {rel_err}')
        
        MAPE_smaple = abs((Htrue-Hest)/Htrue)
        
        return rel_err, MAPE_smaple
                
        
    
    def compute_PWA_error(self, ref_chan, center_freq):
        """
        Compute pwa model error

        Parameters
        ----------
        ref_chan : MPChan
            the reference MPChan object

        Returns
        -------
        PWA complex gain squared error 

        """
        phase_ls = []
        gain_ls = []
        dly_ls = []

        for i_path in range(ref_chan.npath):
            aoa_az = ref_chan.ang[i_path,0]
            aoa_el = ref_chan.ang[i_path,1]
            aod_az = ref_chan.ang[i_path,2]
            aod_el = ref_chan.ang[i_path,3]
            
            # direction of arrival
            u_r = np.array([np.cos(aoa_az)*np.cos(aoa_el), 
                          np.sin(aoa_az)*np.cos(aoa_el),
                          np.sin(aoa_el)]) 
            
            # direction of departure
            u_t = np.array([np.cos(aod_az)*np.cos(aod_el), 
                          np.sin(aod_az)*np.cos(aod_el),
                          np.sin(aod_el)]) 
            
            # pwa approximated distance
            est_dist = PhyConst.light_speed_remcom * ref_chan.dly[i_path] +\
                    u_r.dot(ref_chan.rx_coord - self.rx_coord) +\
                    u_t.dot(ref_chan.tx_coord - self.tx_coord) 
            
            # compute arbitrary phase
            phase_ = 2*np.pi -(2*np.pi*center_freq*ref_chan.dly[i_path])%(2*np.pi)
            arbitrary_phase = np.deg2rad(ref_chan.phase[i_path]) - phase_
            
            phase_from_est_dist = 2*np.pi - (2*np.pi*est_dist*center_freq/PhyConst.light_speed_remcom) % (2*np.pi)
            phase_from_est_dist = phase_from_est_dist + arbitrary_phase
            dly_from_est_dist = est_dist/PhyConst.light_speed_remcom
            
            phase_ls.append(phase_from_est_dist)
            gain_ls.append(-ref_chan.pl[i_path])
            dly_ls.append(dly_from_est_dist)
            
        Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, self.rnd_freq, center_freq)
        Htrue, Etrue = comp_cmplx_gain(-self.pl, np.deg2rad(self.phase), self.dly, self.rnd_freq, center_freq)
        rel_err = np.abs(Hest-Htrue)**2/Eest
        print(f'PWA Rel_err = {rel_err}')
        MAPE_smaple = abs((Htrue-Hest)/Htrue)
        
        return rel_err, MAPE_smaple
    
    
    
    def compute_const_error(self, ref_chan, center_freq):
        """
        Compute constant model error
        
        Constant model is that we directly use ref_channel h as our estmated h

        Parameters
        ----------
        ref_chan : MPChan
            the reference MPChan object

        Returns
        -------
        PWA complex gain squared error 

        """
        Href, Eref = comp_cmplx_gain(-ref_chan.pl, np.deg2rad(ref_chan.phase), ref_chan.dly, ref_chan.rnd_freq, center_freq)
            
        Htrue, Etrue = comp_cmplx_gain(-self.pl, np.deg2rad(self.phase), self.dly, self.rnd_freq, center_freq)
        
        rel_err = np.abs(Href-Htrue)**2/Eref
        print(f'Contant Rel_err = {rel_err}')
        MAPE_smaple = abs((Htrue-Href)/Htrue)
        
        return rel_err, MAPE_smaple
        
    
    def compute_gamma(self, ref_chan):
        """
        Compute gamma angle for all path in this MPChan

        Parameters
        ----------
        ref_chan : MPChan
            the reference MPChan object

        Returns
        -------
        gamma angle list

        """
                
        #TODO: write a function that matching the paths of two MPChan
        self.path_match(ref_chan)
        # inital for gamma result
        self.gamma = np.zeros([len(self.path_map), 2])
        #TODO: write support functions from code we have
        # and solve gamma angles and return
        for i_path in range(self.npath):
            # if find matched reference path
                if self.path_map[i_path] != -1:
                    self.gamma[i_path, 0], self.gamma[i_path, 1] = \
                        self.find_gamma(ref_chan, i_path, self.path_map[i_path])
                    # print(self.gamma[i_path, 0], self.gamma[i_path, 1])
                    if np.isnan(self.gamma[i_path, 0]) or\
                        np.isnan(self.gamma[i_path, 1]) or\
                            np.isinf(self.gamma[i_path, 0]) or\
                                np.isinf(self.gamma[i_path, 1]):
                        self.path_map[i_path] = -1
                        
        
    def path_match(self, mp_chan, TH = 3, if_print = False):
        """
        Path match with another MPChan object.

        Parameters
        ----------
        mp_chan : MPChan object
            Another MPChan. 
            It can be the reference TX-RX channel.
            (for computing gamma angle)
            And it also can be another displaced TX-RX channel 
            (for computing error)
        TH : threshold

        Returns
        -------
        None.
        
        Two new self objects:
            

        """
        
        self.path_map = None # indicate the path number in the another MPChan
        # list indicate that if the path already matched with other
        path_number_used = [0]*mp_chan.npath 
        # draft choice from angles matching 
        # (this is because maybe more paths matched the same path)
        # then we need to match again by delay
        draft_id = []
        
        for i_path in range(self.npath):
            # Match by angles
            angs = self.ang[i_path, :].squeeze() # (4,) path's angles
            # calculate the angles abs difference between this path with 
            # all paths in other MPChan.
            angs_diff = np.sum(np.abs(angs-mp_chan.ang), axis=1)
            # find delay diff
            dly_diff = np.abs(self.dly[i_path] - mp_chan.dly)
            
            idx_by_angs = np.argmin(angs_diff)
            ref_path_interaction = mp_chan.interact[idx_by_angs]
            
            # check if interaction ==
            if (self.interact[i_path] == ref_path_interaction) and\
                (abs(np.argmin(angs_diff) - np.argmin(dly_diff)) < TH):
                # get the id
                draft_id.append(np.argmin(angs_diff)) # path start from 1
            else:
                # append -1
                draft_id.append(-1)
            

        # check if we matched two paths with the same reference tx-rx path
        for i_path in range(self.npath):
            # if -1 means not match any path
            if draft_id[i_path] != -1:
                path_number_used[draft_id[i_path]] += 1

        if any(x > 1 for x in path_number_used):
            print("matched two paths with the same reference tx-rx path")
            # weaker path go -1
            path_number_used = [0]*mp_chan.npath
            for i_path in range(self.npath):
                if draft_id[i_path] != -1:
                    if path_number_used[draft_id[i_path]] == 0:
                        path_number_used[draft_id[i_path]] = 1
                    else:
                        draft_id[i_path] = -1     
        else:
            print("no problem for matching paths")
            if if_print:
                print(draft_id)
                
        self.path_map = draft_id

    
    def find_gamma(self, ref_chan, path_id, ref_path_id):
        """
        Find gamma value for a matched path

        Parameters
        ----------
        ref_chan
        
        path_id
        
        ref_path_id

        Returns
        -------
        Two gamma angles value.

        """
        # aoa and aod
        aoa_az = ref_chan.ang[ref_path_id, 0]
        aoa_el = ref_chan.ang[ref_path_id, 1]
        aod_az = ref_chan.ang[ref_path_id, 2]
        aod_el = ref_chan.ang[ref_path_id, 3]
        
        # coordinate
        ref_chan.tx_coord = ref_chan.tx_coord
        displaced_tx = self.tx_coord
        ref_chan.rx_coord = ref_chan.rx_coord
        displaced_rx = self.rx_coord
        
        # eqn left
        eqn_left = self.dly[path_id] * PhyConst.light_speed_remcom
        
        # calculate right hand
        # c*tau*e_x
        ref_path_distance = ref_chan.dly[ref_path_id] * PhyConst.light_speed_remcom
        item_1 = np.array([ref_path_distance, 0, 0])
        
        # Ry(aoa_el)*Rz(-aoa_az)*(x^r_0-x^r)
        item_2 = Ry(aoa_el).dot(Rz(-aoa_az)).dot(ref_chan.rx_coord-displaced_rx)
        #######################################################################
        # solve if s = -1:
        s = -1 
        # solve gamma
        gamma_1 = scipy.optimize.root_scalar(scipy_f, x0=0, x1=np.pi, args=(item_1, item_2, eqn_left, s, aod_el, aod_az, ref_chan.tx_coord, displaced_tx))
        #######################################################################
        # solve if s = +1:
        s = 1 
        
        gamma_2 = scipy.optimize.root_scalar(scipy_f, x0=0, x1=np.pi, args=(item_1, item_2, eqn_left, s, aod_el, aod_az, ref_chan.tx_coord, displaced_tx))
        
        return gamma_1.root, gamma_2.root



def scipy_f(x, item_1, item_2, eqn_left, s, aod_el, aod_az, ref_chan_tx_coord, displaced_tx):
    """
    Function use to solve gamma value

    Returns
    -------
    Scipy eqauation.

    """
    
    item_3 = Qz(s).dot(Rx(x)).dot(Ry(aod_el)).dot(Rz(-aod_az)).dot(ref_chan_tx_coord-displaced_tx)
    eq_right_vector = item_1 + item_2 + item_3
    eq_right_power2 = np.sum(eq_right_vector**2)
    
    return eq_right_power2 - eqn_left**2


def comp_cmplx_gain(gain, phase, delay, rnd_freq, center_freq):
        
    """
    Calculate complex channel gain and Eavg
    
    Returns
    -------
    complex channel gain
    Eavg 
    """
    h = 0
    Eavg = 0
    delta = rnd_freq-center_freq
    for i in range(len(gain)):
        g = np.sqrt(10 ** (gain[i]/10)) # convert dB to linear
        h += g * np.exp((phase[i]+2*np.pi*delta*delay[i])*1j) 
        Eavg += g**2

    return h, Eavg
            
       
def test_channl(df_root, ref_df_root, test_df_root, test_channel_num, i_lamdba, results, mape_results, test_freq, i_freq, center_freq):
    
    """
    Test inital MPChan object
    
    Parameters
    ----------
    df_root : string
        dataframe file root of a displacement MPChan that using for compute gamma
    ref_df_root : string
        dataframe file root of a reference MPChan
    test_df_root : string
        dataframe file root of a displacement MPChan for validation
        
    test_channel_num : int
        number of link which tests.
    i_lamdba : int
        indicate the lamdba id of lamdba list
    results : np.array
        store the results

    Returns
    -------
    None.

    """
    
    # load pd data frame
    df = pd.read_csv(df_root)
    channel_df = df.iloc[[test_channel_num]]
    channel_df = channel_df.reset_index()
    
    ref_df = pd.read_csv(ref_df_root)
    ref_channel_df = ref_df.iloc[[test_channel_num]]
    ref_channel_df = ref_channel_df.reset_index()
    
    test_df = pd.read_csv(test_df_root)
    test_channel_df = test_df.iloc[[test_channel_num]]
    test_channel_df = test_channel_df.reset_index()
        
    ##### Test path match
    if np.isnan(channel_df.at[0, DataConfig.paths_number]):
        mp_chan = MPChan(channel_df,test_freq,True)
        print("displaced pair outage")
    else:
        mp_chan = MPChan(channel_df,test_freq)
        if np.isnan(ref_channel_df.at[0, DataConfig.paths_number]):
            print("reference outage")
        else:
            ref_chan = MPChan(ref_channel_df,test_freq)
            if np.isnan(test_channel_df.at[0, DataConfig.paths_number]):
                print("test outage")
            else:
                test_chan = MPChan(test_channel_df,test_freq)
                relerr_rm, mape_rm = mp_chan.compute_RM_error(ref_chan, test_chan,center_freq)
                relerr_pwa, mape_pwa = test_chan.compute_PWA_error(ref_chan,center_freq)
                relerr_const, mape_const = test_chan.compute_const_error(ref_chan, center_freq)
                    
                results[i_freq, i_lamdba, test_channel_num, 0] = relerr_pwa
                results[i_freq, i_lamdba, test_channel_num, 1] = relerr_rm
                results[i_freq, i_lamdba, test_channel_num, 2] = relerr_const
                
                mape_results[i_freq, i_lamdba, test_channel_num, 0] = mape_pwa
                mape_results[i_freq, i_lamdba, test_channel_num, 1] = mape_rm
                mape_results[i_freq, i_lamdba, test_channel_num, 2] = mape_const
                               

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Validate reflection model')
parser.add_argument('--center_freq',action='store',default=28e9, type=float,
                    help='Center frequency')
parser.add_argument('--case_name',action='store',default= 'no_foliage_no_diffraction', 
                    help='indicate if includes foliage and diffraction')
parser.add_argument('--n_freq',action='store',default=10,type=int,
                    help='number of random frequencies within the bandwidth')

args = parser.parse_args()
center_freq = args.center_freq
case_name = args.case_name
n_freq = args.n_freq # the number of random frequencies around center frequency

str_freq = str(int(center_freq/1e9))
df_root = "./ray_tracing_data/"+str_freq+'GHz/'+case_name+"/Beijing_1.0_fix.csv"
ref_df_root = "./ray_tracing_data/"+str_freq+'GHz/'+case_name+"/Beijing_ref_fix.csv"
lambda_value = ['2.0', '5.0', '10.0', '50.0', '100.0']
n_test = 43 # the number of Tx-Rx pairs
rnd_freq_arr = np.array(random.sample(range(int(center_freq-1e9), int(center_freq+1e9)), n_freq), dtype=float)

# 0 for PWA, 1 for RM, 2 for Constant Model
results = np.empty([n_freq, len(lambda_value), n_test, 3]) 
results[:] = np.nan
mape_results = np.empty([n_freq, len(lambda_value), n_test, 3]) 
mape_results[:] = np.nan

for ifc, fc in enumerate(rnd_freq_arr):
    for i in range(len(lambda_value)):
        test_df_root = "./ray_tracing_data/"+str_freq+'GHz/'+case_name+"/Beijing_" + lambda_value[i] +"_fix.csv"
        for i_test in range(n_test):
            test_channel_num = int(i_test)
            print(f'{fc}:{lambda_value[i]}:{i_test}')
            test_channl(df_root, ref_df_root, test_df_root, i_test, i, results, mape_results, fc, ifc, center_freq)   

# Save as CSV
df = pd.DataFrame(columns = ['frequency','lambda_dist','link_id','RM_error','PWA_error','ConstM_error'])
lambda_dist_ls = ['2.0', '5.0', '10.0', '50.0', '100.0']
n_lambda = 5
for ifreq in range(n_freq):
    for ilambda in range(n_lambda):
        for itest in range(n_test):
            df = df.append({'frequency': rnd_freq_arr[ifreq],\
                            'lambda_dist': lambda_dist_ls[ilambda],\
                            'link_id': itest, \
                           'PWA_error' : results[ifreq, ilambda, itest, 0],\
                           'RM_error' : results[ifreq, ilambda, itest, 1],\
                           'ConstM_error' : results[ifreq, ilambda, itest, 2],\
                            'PWA_mape_sample' : mape_results[ifreq, ilambda, itest, 0],\
                            'RM_mape_sample' : mape_results[ifreq, ilambda, itest, 1],\
                            'ConstM_mape_sample' : mape_results[ifreq, ilambda, itest, 2],\
                               }, ignore_index=True)

df.to_csv('results/'+str_freq+'GHz/err_results_'+ case_name +'.csv',index = False)       