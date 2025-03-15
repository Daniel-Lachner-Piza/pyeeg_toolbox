import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mne

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
                output_path: Path=None,
                file_extension: str='.lay')->None:
        self.eeg_data_path = eeg_data_path
        self.ch_coordinates_data_path = ch_coordinates_data_path
        self.szr_info_data_path = szr_info_data_path
        self.output_path = output_path
        self.eeg_files_ls = get_files_in_folder(eeg_data_path, file_extension)

    def get_hourly_ni(self, pat_id: str=None, mtg_t:str='ir')->None:

        eeg_type = 'ieeg'
        is_ieeg = True
        if mtg_t == 'sr' or mtg_t == 'sb':
            eeg_type = 'scalp_eeg'
            is_ieeg = False

        ni_df_fname = self.output_path / f'{pat_id}_ni_hourly{eeg_type}.csv'

        if not ni_df_fname.exists():
            ni_df = pd.DataFrame()
            for this_pat_eeg_fpath in self.eeg_files_ls:
                print(this_pat_eeg_fpath.name)
                eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t=mtg_t)
                eeg_reader.remove_natus_virtual_channels()

                for start_sample in np.arange(0, eeg_reader.n_samples, 3600*eeg_reader.fs, dtype=int):
                    end_sample = int(start_sample + 3600*eeg_reader.fs)
                    if end_sample > eeg_reader.n_samples:
                        end_sample = eeg_reader.n_samples
                        pass
                    print((start_sample, end_sample))
                    hr_nr = int(np.ceil(start_sample/(3600*eeg_reader.fs)))
                    all_ch_sigs = eeg_reader.get_data(start=start_sample, stop=end_sample)
                    ni_tvec, chspec_ni_vec, chavg_ni_vec = get_noise_index_vec(fs=eeg_reader.fs, mtg_labels=eeg_reader.ch_names, mtg_signals=all_ch_sigs, notched=False, is_ieeg=is_ieeg)
                    ch_spec_avg_ni = np.mean(chspec_ni_vec, axis=1)
                    print(this_pat_eeg_fpath.name)
                    this_hour_ni_df = pd.DataFrame(data={'eeg_file': [this_pat_eeg_fpath.name]*len(ch_spec_avg_ni), 'hour': [hr_nr]*len(ch_spec_avg_ni), 'mtg_name': eeg_reader.ch_names, 'ch_spec_avg_ni':ch_spec_avg_ni})
                    ni_df = pd.concat([ni_df, this_hour_ni_df])
                    pass
            ni_df.to_csv(ni_df_fname, index=False)
        else:
            ni_df = pd.read_csv(ni_df_fname)
        return ni_df

    def clean_data(self, pat_id: str=None, mtg_t:str='ir')->None:
                
        for this_pat_eeg_fpath in self.eeg_files_ls:
            print(this_pat_eeg_fpath.name)

            ieeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t='ir')
            ieeg_reader.remove_natus_virtual_channels()
            intracr_ni_df = self.get_hourly_ni(pat_id=pat_id, mtg_t='ir')

            scalp_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t='sr')
            scalp_reader.remove_natus_virtual_channels()
            scalp_ni_df = self.get_hourly_ni(pat_id=pat_id, mtg_t='sr')

            keep_channels_ls = ieeg_reader.ch_names
            keep_channels_ls.extend(scalp_reader.ch_names)
            keep_channels_ls = [str(ch) for ch in np.unique(keep_channels_ls)]

            eeg_data = mne.io.read_raw_persyst(this_pat_eeg_fpath, verbose=False)
            eeg_data.info['subject_info']['birthday'] = pd.Timestamp(eeg_data.info['subject_info']['birthday'])
            #birthday = ieeg_data.info['subject_info']['birthday']
            #ieeg_data.info['subject_info']['birthday'] = [birthday.year, birthday.month, birthday.day]
            time = eeg_data.times
            nr_samples = eeg_data.n_times
            fs = eeg_data.info["sfreq"]
            for start_time in np.arange(0, time[-1], 3600, dtype=int):
                end_time = start_time + 3600
                if end_time > time[-1]:
                    end_time = time[-1]

                hr_nr = int(np.ceil(start_time/(3600)))

                eeg_data.pick(keep_channels_ls)
                eeg_data.crop(start_time, end_time, include_tmax=False, verbose=True)


                out_dir= f"{self.output_path}/ConvertedEEG/{pat_id}/"

                out_raw_ieeg = out_dir+str(this_pat_eeg_fpath.name).replace('.lay', f'_h{hr_nr:02d}.edf')
                os.makedirs(Path(out_raw_ieeg).parent, exist_ok=True)

                try:
                    eeg_data.export(fname=out_raw_ieeg, fmt='edf', overwrite=True, verbose=True)
                    #ieeg_data.save(fname=out_raw_ieeg, picks=eeg_reader.ch_indices, tmin=start_time, tmax=end_time, fmt='single', overwrite=True, split_size='2GB', split_naming='bids', verbose=None)
                except:
                    print("Saving Raw file failed")
                    os.remove(out_raw_ieeg)
                ################
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