import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO
from studies_info import ACH_Pediatric_Patients
from pyeeg_toolbox.utils.io_tools import get_files_in_folder
from pyeeg_toolbox.dsp.noise_index import get_noise_index_vec

class Clean_EEG_Data:
    def __init__(self, pat_id:str=None, 
                eeg_data_path: Path=None,
                ch_coordinates_data_path: Path=None, 
                szr_info_data_path: Path=None, 
                output_path: Path=None)->None:
        self.eeg_data_path = eeg_data_path
        self.ch_coordinates_data_path = ch_coordinates_data_path
        self.szr_info_data_path = szr_info_data_path
        self.output_path = output_path

    def clean_data(self, pat_id: str=None, mtg_t:str='ir')->None:
        eeg_files_ls = get_files_in_folder(ieeg_data_path=self.eeg_data_path, file_extension='.lay')

        for this_pat_eeg_fpath in eeg_files_ls:
            print(this_pat_eeg_fpath.name)
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t=mtg_t)
            eeg_reader.remove_natus_virtual_channels()

            start_sample = 0
            end_sample = int(np.round(eeg_reader.fs*60*10))
            all_ch_sigs = eeg_reader.get_data(start=start_sample, stop=end_sample)
            ni_tvec, chspec_ni_vec, chavg_ni_vec = get_noise_index_vec(fs=eeg_reader.fs, mtg_labels=eeg_reader.ch_names, mtg_signals=all_ch_sigs, notched=False, is_ieeg=True)
            ch_spec_avg_ni = np.mean(chspec_ni_vec, axis=1)
            print(this_pat_eeg_fpath.name)
            df = pd.DataFrame(data={'mtg_name': eeg_reader.ch_names, 'ch_spec_avg_ni':ch_spec_avg_ni})
            pass
        pass


if __name__ == "__main__":
    output_path = Path(os.getcwd()) / "Output"
    os.makedirs(output_path, exist_ok=True)
    study_info = ACH_Pediatric_Patients()
    for pat_id in study_info.patients.keys():

        print(pat_id)
        ieeg_data_path = study_info.eeg_data_path / pat_id
        ch_coordinates_data_path = study_info.channel_coordinates_data_path
        szr_info_data_path = study_info.seizure_info_data_path
        output_path=output_path

        cleaner = Clean_EEG_Data(pat_id=pat_id, eeg_data_path=ieeg_data_path, ch_coordinates_data_path=ch_coordinates_data_path, szr_info_data_path=szr_info_data_path, output_path=output_path)
        cleaner.clean_data(pat_id=pat_id, mtg_t='ir')
    pass