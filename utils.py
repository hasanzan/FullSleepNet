import mne
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import h5py


def read_shh1_data(data_path, label_path) -> tuple:

    """
    Load .edf data and arousal locations

    Parameters
    -----------
        data_path : path to edf data
        label_path: path to locations

    Returns
    ----------
        a tuple containing a ndarray of edf data and 
        an ndarray of arousals containing zeros and ones
    """

    # channel to exclude
    exc = [
        'SaO2', 'H.R.', 'SOUND','AIRFLOW', 'RES', 
        'THOR RES', 'ABDO RES', 'POSITION','LIGHT',  
        'NEW AIR', 'OX stat', 'stat', 'EPMS', 'AIR', 'AUX',
        'EOG(L)', 'EOG(R)', 'ECG', 'EEG(sec)'
    ]

    # read raw data
    raw_data = mne.io.read_raw_edf(data_path, verbose=0,
                                  exclude=['SaO2', 'H.R.', 'ECG', 'SOUND', 
                                           'AIRFLOW', 'RES', 'THOR RES', 'ABDO RES', 'POSITION', 
                                           'LIGHT', 'NEW AIR', 'OX stat', 'stat', 'EPMS', 'AIR', 'AUX'],
                                  infer_types=False)
    
    # get ch names   
    raw_data.pick_channels(['EEG'], ordered=True)

    # sampling frequency
    fs = raw_data.info["sfreq"]

    # read data
    data = raw_data.get_data().T
    
    # standardize
    mu = data.mean(axis=0, keepdims=True)
    sigma = data.std(axis=0, keepdims=True)
    data = (data - mu) / sigma

    # read stage labels
    tree = ET.parse(label_path)
    root = tree.getroot()
    stages = np.array([stage.text for stage in root[4].findall("SleepStage")], dtype=np.int8)
    
    # merge stage 3 and 4, change 5 to 4
    stages[stages == 4] = 3
    stages[stages == 5] = 4
    stages[stages > 5] = 5


    # arousal start and end time
    arousals = np.array([[float(event.find("Start").text), float(event.find("Duration").text)] for event in root[3].findall("ScoredEvent") if event.find("Name").text == "Arousal ()"])
    
    if len(arousals) == 0:
        return data, np.zeros(shape=(data.shape[0],), dtype=np.int64), np.repeat(stages, 30 * fs) 
    
    # convert to start sample stop sample
    arousals = arousals * fs
    arousals[:,1] = arousals[:,0] + arousals[:,1]
    arousals = arousals.astype(int)

    # convert to zeros and ones
    arousals_bin = np.zeros(shape=(data.shape[0],), dtype=np.int64)
    for start_idx, end_idx in arousals:
        arousals_bin[start_idx:end_idx+1] = 1
    
   
    # return data and stages
    return data, arousals_bin, np.repeat(stages, 30 * fs)
   
def read_mesa_data(data_path, label_path) -> tuple:

    """
    Load .edf data and arousal locations

    Parameters
    -----------
        data_path: path to edf data
        label_path: path to arousals

    Returns
    ----------
        a tuple containing a ndarray of edf data and 
        an ndarray of arousals containing zeros and ones
    """

    # channel to exclude
    exc = ['EKG', 'EOG-L', 'EOG-R', 'EMG', 'EEG1', 'EEG2', 'Pres', 'Flow', 'Snore', 'Thor', 'Abdo', 
            'Leg', 'Therm', 'Pos', 'EKG_Off', 'EOG-L_Off', 'EOG-R_Off', 'EMG_Off', 'EEG1_Off', 'EEG2_Off',
            'EEG3_Off', 'Pleth', 'OxStatus', 'SpO2', 'HR', 'DHR']

    # read edf file
    raw_edf = mne.io.read_raw_edf(data_path, exclude=exc, infer_types=False, verbose=0)

    # channel names
    ch_names = raw_edf.info.ch_names

    # pick channels
    new_ch = ["EEG3"]
    raw_edf.pick_channels(new_ch, ordered=True)

    # sampling freq
    fs = raw_edf.info["sfreq"]

    # read data
    data = raw_edf.get_data().T    

    # read stage labels
    tree = ET.parse(label_path)
    root = tree.getroot()
    stages = np.array([stage.text for stage in root[4].findall("SleepStage")], dtype=np.int8)

    # merge stage 3 and 4, change 5 to 4
    stages[stages == 4] = 3
    stages[stages == 5] = 4
    stages[stages > 5] = 5

    # pad stages
    stages = np.repeat(stages, fs*30)
#     stages = np.pad(stages, pad_width=((0, data.shape[0] - stages.shape[0])))
    data = data[:stages.shape[0]]
   
    # read arousal arousals
    # read tree
    tree = ET.parse(label_path)
    root = tree.getroot()

    # arousal start and end time
    arousals = np.array([[float(event.find("Start").text), float(event.find("Duration").text)] for event in root[3].findall("ScoredEvent") if event.find("Name").text == "Arousal ()"])
    
    if len(arousals) != 0:

        # convert to start sample stop sample
        arousals = arousals * fs
        arousals[:,1] = arousals[:,0] + arousals[:,1]
        arousals = arousals.astype(int)

        # convert to zeros and ones
        arousals_bin = np.zeros(shape=(data.shape[0],), dtype=np.int64)
        for start_idx, end_idx in arousals:
            arousals_bin[start_idx:end_idx+1] = 1
    else:
        arousals_bin = np.zeros(shape=(stages.shape[0],), dtype=np.int64)
    
    # allow just 30 minutes before and after sleep
    c = (stages >= 1) & (stages <= 5)
    first_sleep, last_sleep = np.flatnonzero(c)[[0, -1]]
    
    # thirty min
    thirty_min = fs * 60 * 30
    
    # trim from left and rigth
    start_idx = first_sleep - thirty_min if first_sleep > thirty_min else 0
    end_idx = last_sleep + thirty_min if last_sleep + thirty_min < stages.shape[0] else -1
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    
    data = data[start_idx:end_idx]
    arousals_bin = arousals_bin[start_idx:end_idx]
    stages = stages[start_idx:end_idx]
        
    
    # resample and standardize
    data = data[::2]
    mu = data.mean(axis=0, keepdims=True)
    sigma = data.std(axis=0, keepdims=True)
    data = (data - mu) / sigma


    return data.astype(np.float32), arousals_bin, stages

def set_random_seed(seed):
    """Sets all random seeds for the program (Python, NumPy, and TensorFlow).
    
    ```python
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ```
    Arguments:
      seed: Integer, the random seed to use.
    """
    import random
    import tensorflow as tf
    import os
    
    if not isinstance(seed, int):
        raise ValueError(
            "Expected `seed` argument to be an integer. "
            f"Received: seed={seed} (of type {type(seed)})"
        )
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)