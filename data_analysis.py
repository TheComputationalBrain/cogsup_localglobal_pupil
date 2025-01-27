import glob
import os
import re
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eyetracker_tools.preprocess_eyedata import pupil


BLINK_EXCLUSION_RATIO = 0.2
DECIMATE_FACTOR = 5


def process_eyelink_data(eyelink_list,
                         window,
                         baseline,
                         reject_epoch=True,
                         plot=False,
                         zscore=True):
    """
    Preprocess the eyetracker datafile.

    Parameters
    ----------
    eyelink_list : list
        list of file names
    window : list
        interval [start, stop] in s, defining the epoch relative
        to the event of interest.
    reject_epoch : boolean, optional
        whether epochs with a lot of blink are rejected. default: True
    plot : boolean, optional
        whether to plot the signal. default: False
    zscore : boolean, optional
        whether to zscore the signal. default: True

    Returns
    -------
    dictionary
        'data': a 2D array epochs x time samples
        'info': a panda dataframe with information about the epochs
        'times': peristimulus times within the epoch (in s)
        'SOA': stimulus onset asynchrony (in s)
    """
    
    cat_data, cat_info = [], []
    
    for file in eyelink_list:
               
        # Import and preprocess pupil data
        pup = pupil()
        pup.import_data(filename=file, col=[2], col_name=['diameter'], eyetracker='eyelink')
        pup.preprocess(plot=plot, eyetracker='eyelink')
        
        # Get event info
        stim_onset = [msg['time'] for msg in pup.events if re.match('block', msg['msg'])]
        sess_num = [eyelink_list.index(file)+1]*len(stim_onset)
        block_num = [msg['msg'].split(',')[0].strip('block ')
                     for msg in pup.events if re.match('block', msg['msg'])]
        stim_num = [int(msg['msg'].split(',')[1].strip('trial '))
                    for msg in pup.events if re.match('block', msg['msg'])]
        stim_type = [msg['msg'].split(',')[2].strip('stim ')
                     for msg in pup.events if re.match('block', msg['msg'])]
        
        # Plot and save quality check plots   
        plt.xlabel("Time stamp", fontsize=14)
        plt.ylabel("Pupil diameter", fontsize=16)
        plt.title(f"Session {sess_num[0]}", fontsize=18)    
        
        # Epoch data
        if zscore:
            pup.data['diameter_int'] = scipy.stats.zscore(pup.data['diameter_int'])
        pup.epoch_eyelink(stim_onset, variable='diameter_int', decimate=DECIMATE_FACTOR,
                  before=-window[0], after=window[1], baseline=baseline,
                  reject=reject_epoch, blink_ratio_thd=BLINK_EXCLUSION_RATIO,
                  conditions={'session': np.array(sess_num),
                              'block': np.array(block_num, dtype=int),
                              'stim': np.array(stim_type),
                              'stim_num': np.array(stim_num, dtype=int)})

        cat_data.append(pup.epochs)
        cat_info.append(pup.epochs_info)

    return {'data': np.vstack(cat_data), 'info': pd.concat(cat_info),
            'times': pup.epochs_times, 'SOA': np.median(np.diff(stim_onset))}


if __name__ == '__main__':
    # get data files
    data_files = sorted(glob.glob(os.path.join('example_data_set', '*.asc')))
            
    ## Process data 
    data = process_eyelink_data(data_files, baseline=[-0.2, 0], window=[-0.25, 3],
                                reject_epoch=True, plot=True)

    # recode stimulus as LSGS, LSGD, LDGS, LDGD
    data['info']['stim_type'] = None
    for sess_num in data['info']['session'].unique():
        for block_num in data['info']['block'].unique():

            stim_selection = (data['info']['session'] == sess_num) & (data['info']['block'] == block_num)
            event = data['info'][stim_selection]
            event = event.reset_index()
            
            if event['stim'].to_list()[0] == 'AAAAA': # define the frequent stimulus
                data['info'].loc[stim_selection, 'stim_type'] = np.where(event['stim']== 'AAAAA', 'LSGS', 'LDGD')

            elif event['stim'].to_list()[0] == 'BBBBB': # define the frequent stimulus
                data['info'].loc[stim_selection, 'stim_type'] = np.where(event['stim']== 'BBBBB', 'LSGS', 'LDGD')

            elif event['stim'].to_list()[0] == 'AAAAB': # define the frequent stimulus
                data['info'].loc[stim_selection, 'stim_type'] = np.where(event['stim']== 'AAAAB', 'LDGS', 'LSGD')

            elif event['stim'].to_list()[0] == 'BBBBA': # define the frequent stimulus
                data['info'].loc[stim_selection, 'stim_type'] = np.where(event['stim']== 'BBBBA', 'LDGS', 'LSGD')

    line_style = {'LSGS': '--',
                'LDGS': '--',
                'LSGD': '-',
                'LDGD': '-',}
    line_color = {'LSGS': 'blue',
                'LDGS': 'orange',
                'LSGD': 'blue',
                'LDGD': 'orange',}

    plt.figure()
    for condition in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
            erp_mean = np.mean(data['data'][data['info']['stim_type'] == condition, :], axis=0)
            erp_sem = scipy.stats.sem(data['data'][data['info']['stim_type'] == condition, :], axis=0)
            plt.plot(data['times'],
                    erp_mean,
                    label=condition, color=line_color[condition], linestyle=line_style[condition])
            plt.fill_between(data['times'],
                            (erp_mean-erp_sem), (erp_mean+erp_sem),
                            color=line_color[condition], alpha=.1)
    plt.xlabel('Peristimulus time (s)')
    plt.ylabel('Pupil size (z units)')
    plt.legend()
    plt.show()

