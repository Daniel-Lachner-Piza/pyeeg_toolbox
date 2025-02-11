import mne
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


from pathlib import Path
#from micromed_io.to_mne import create_mne_from_micromed_recording
from micromed_io.trc import MicromedTRC

ieeg_data_path = Path("D:/")

file_extension = '.trc'
pat_files_ls = [fn for fn in ieeg_data_path.rglob(f"*{file_extension}")]


pats_sampling_rates_ls = defaultdict(list)
for eeg_filepath in pat_files_ls:
    print(eeg_filepath)
    mmtrc  = MicromedTRC(eeg_filepath)
    eeg_hdr = mmtrc.get_header()
    eeg_hdr.name
    pats_sampling_rates_ls[eeg_hdr.name].append(eeg_hdr.min_sampling_rate)
    pass


fs_description = {"Filepath":[], "ID":[], "NrRecords":[], "NrRecordsAbove1kHz":[]}
fi = 0
for k,v in pats_sampling_rates_ls.items():
    nr_records = len(v)
    nr_records_above_1khz = np.sum(np.array(v) > 1000)
    fs_description["Filepath"].append(k)
    fs_description["ID"].append(k)
    fs_description["NrRecords"].append(nr_records)
    fs_description["NrRecordsAbove1kHz"].append(nr_records_above_1khz)
pass