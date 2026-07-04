# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np
import socket
import struct
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne
from pylsl import StreamInfo, StreamOutlet
from scipy import signal
from metabci.brainflow.amplifiers import Marker,DitingBrainEEGAmplifier
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.decomposition import FBSCCA
from sklearn.base import BaseEstimator, ClassifierMixin
cre_list = []


'''-----------------------In[5.1]数据滤波处理--------------------------'''
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    srate = 1000
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    sig_new = signal.filtfilt(b_notch, a_notch, sig)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def cca_train_model():
    
    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]
    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 1000)

    model = FBSCCA(filterbank=filterbank, n_components=5, filterweights=filterweights, n_jobs=-1)

    return model 


class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, event_map,
                 srate, lsl_source_id, timeout, worker_name, ch_ind):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = event_map
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.data_matlist = []
        self.mode = None
        self.ch_ind = ch_ind
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        self.model = cca_train_model() 

        info = StreamInfo(
            name='meta_online_worker',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')


    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        print(data.shape)
        data = data[self.ch_ind]
        f0 = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=1000)
        data = signal.filtfilt(b_notch, a_notch, data)

        freq_list = [8, 9, 10, 11, 12, 13]

        Yf = generate_cca_references(freq_list, srate=1000, T=2, n_harmonics = 5)
        self.model.fit(data,Yf=Yf)
        p_labels_cca = self.model.predict(data)
        p_labels_cca = p_labels_cca+1

        print("p_labels:", p_labels_cca)

        k = 0
        my_list = [[1], [2], [3], [4], [5], [6]]
        cre_list.append(p_labels_cca+1)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 4]:
                k = k + 1
        acc = float(k) / float(len(cre_list))
        print("在线识别准确率：", acc)

        data = struct.pack('!i', p_labels_cca[0])

        # LSL 
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels_cca)

    def post(self):
        try:
            self.udp_sock.close()
        except Exception:
            pass


if __name__ == '__main__':

    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 2.14]
    # Label types
    stim_labels = list(range(1, 7))
    event_map = {str(e):e for e in range(1, 255)}
    # stim_labels = 8
    cnts = 1
    # Data path
    run_files = ['./Data/Tju-testOffline/1130/S2.bdf']
    run_files = ['C:\\Users\\DELL\\Desktop\\移动\\0408\\offline2.bdf']
    pick_chs = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O2', 'O1']
    ch_ind= np.array([1,2,3,4,5,6,7,8,9,10,11,12],dtype=int)-1  # 真实选择导联

    lsl_source_id = 'meta_online_worker666'
    feedback_worker_name = 'feedback_worker'
    udp_target = ('172.21.20.107', 7810)

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        event_map=event_map, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name, ch_ind = ch_ind)
    marker = Marker(stim_interval,srate,stim_labels)

    # worker.pre()
    # worker.consume(marker.get_epoch())
    # # Set Neuroscan parameters
    
    ns = DitingBrainEEGAmplifier()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()

