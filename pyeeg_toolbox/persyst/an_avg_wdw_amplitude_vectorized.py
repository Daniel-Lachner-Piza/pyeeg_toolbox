import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import plotly.colors

from plotly.subplots import make_subplots
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict
from pyeeg_toolbox.persyst.avg_wdw_cumulator import AvgWdwCumulator
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO
from pyeeg_toolbox.dsp.noise_index import get_noise_index_vec
from scipy.signal import find_peaks, peak_prominences
from studies_info import fr_four_patients
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from pyeeg_toolbox.utils.convert_mapped_channels import correct_relabelled_chnames
from statsmodels.stats.multivariate import test_mvmean_2indep
from hotelling.stats import hotelling_t2

# Workaround for Kaleido on Windows, without this the Kaleido executable is not found when saving as .png or .jpg
if sys.platform.startswith('win'):
	os.environ["PATH"] = os.environ["PATH"] + "C:\\Users\\HFO\\Development\\pyeeg_toolbox\\.venv\\Lib\\site-packages\\kaleido\\executable"

class VectorizedAvgWdwAnalyzer(SpikeAmplitudeAnalyzer):
    """
    A class that analyzes for each channel, the averaged time windows that coincide with a spike that was detected on any channel.
    """

    def __init__(self, 
                 pat_id:str=None,
                 ieeg_data_path:str=None, 
                 sleep_data_path:str=None, 
                 ispikes_data_path:str=None,
                 ch_coordinates_data_path:str=None,
                 szr_info_data_path:str=None,
                 sleep_stages_map:Dict[int, str]={0: "Unknown", 1: "N3", 2: "N2", 3:"N1", 4:"REM", 5:"Wake", 6:"NaN"},
                 output_path:Path=None,
                 )->None:
        """
        Initialize the SpikeAmplitudeAnalyzer class.

        Args:
            pat_id (str): The ID of the patient.
            ieeg_data_path (str): The path to the iEEG data files.
            sleep_data_path (str): The path to the sleep stage data files.
            ispikes_data_path (str): The path to the iSpikes data files.
            sleep_stages_map (Dict[int, str]): A dictionary mapping sleep stage codes to their names.

        Returns:
            None
        """
        self.pat_id = pat_id
        self.ieeg_data_path = ieeg_data_path
        self.sleep_data_path = sleep_data_path
        self.ispikes_data_path = ispikes_data_path
        self.ch_coordinates_data_path = ch_coordinates_data_path
        self.szr_info_data_path = szr_info_data_path
        self.sleep_stages_map = sleep_stages_map
        self.output_path = output_path

        if self.sleep_data_path is None:
            self.sleep_data_path = self.ieeg_data_path
        if self.ispikes_data_path is None:
            self.ispikes_data_path = self.ieeg_data_path

        self.eeg_file_extension = ".lay"
        self.pat_files_ls = None
        self.spike_cumulator = None

        super().__init__(
            pat_id=self.pat_id,
            ieeg_data_path=self.ieeg_data_path, 
            sleep_data_path=self.sleep_data_path, 
            ispikes_data_path=self.ispikes_data_path, 
            sleep_stages_map=self.sleep_stages_map,
        )

    def is_natus_virtual_channel(self, mtg_name:str=None) -> bool:
        
        accepted_channs = ["c3", "c4", "cz"]
        non_accepted_hw_groups = ["c", "dc"]
        mtg_name = mtg_name.lower()
        if len(mtg_name.split('-'))>1:
            # Bipolar montage
            mtg_ch_a = mtg_name.split('-')[0]
            mtg_ch_b = mtg_name.split('-')[1]
            hw_group_a = ''.join([c for c in mtg_ch_a if not c.isdigit()])
            hw_group_b = ''.join([c for c in mtg_ch_b if not c.isdigit()])
            if (hw_group_a in non_accepted_hw_groups) or (hw_group_b in non_accepted_hw_groups):
                if (mtg_ch_a not in accepted_channs and mtg_ch_b not in accepted_channs):
                    print(f"Exclude Natus Virtual Channel: {mtg_name}")
                    return True
        else:
            # Referential montage
            hw_group = ''.join([c for c in mtg_name if not c.isdigit()])
            contact_nr = ''.join([c for c in mtg_name if c.isdigit()])
            if (hw_group in non_accepted_hw_groups or len(contact_nr)==0):
                if mtg_name not in accepted_channs:
                    print(f"Exclude Natus Virtual Channel: {mtg_name}")
                    return True

        return False
    
    def summarize_patients_info(self, file_extension:str='.lay', mtg_t:str='ir', force_recalc:bool=False)->None:
        """
        This function summarizes the patient information, including the number of patients, the number of EEG files per patient, and the total duration of EEG data.

        Returns:
        None
        """
        self.get_files_in_folder(file_extension)

        assert len(self.pat_files_ls) > 0, f"No EEG files found in folder {self.ieeg_data_path}"
        #assert len(self.pat_files_ls) >= 40, f"Duration of EEG data is less than 48 hours for patient {self.pat_id}"

        self.pat_files_ls = np.sort(self.pat_files_ls)

        rec_start_idx = 0
        rec_end_idx = 52#48
        if len(self.pat_files_ls) < rec_end_idx:
            ##rec_start_idx = 0
            rec_end_idx = len(self.pat_files_ls)+1
        self.pat_files_ls = self.pat_files_ls[rec_start_idx:rec_end_idx]
        #self.pat_files_ls = self.pat_files_ls[0:52]

        total_eeg_dur_hrs = 0
        nr_ieeg_channs = 0
        sampling_rate = 0
        nr_seizures = 0
        pat_info = {'PatID': [], 'DurationHrs': [], 'NrIEEGChanns': [], 'SamplingRateHz': [], 'NrSeizures': []}
        for this_eeg_fpath in self.pat_files_ls:
            eeg_reader = EEG_IO(eeg_filepath=this_eeg_fpath, mtg_t=mtg_t)
            nr_ieeg_channs = len(eeg_reader.ch_names)
            sampling_rate = eeg_reader.fs
            eeg_dur_hrs = eeg_reader.n_samples / eeg_reader.fs / 3600
            total_eeg_dur_hrs += eeg_dur_hrs
            pass
        pat_info['PatID'].append(self.pat_id)
        pat_info['DurationHrs'].append(int(np.round(total_eeg_dur_hrs)))
        pat_info['NrIEEGChanns'].append(nr_ieeg_channs)
        pat_info['SamplingRateHz'].append(sampling_rate)
        pat_info['NrSeizures'].append(nr_seizures)
        pat_info_df = pd.DataFrame(pat_info)
        os.makedirs(self.output_path / "Patient_Info", exist_ok=True)
        pat_info_fn = self.output_path / "Patient_Info" / f"AllPatsInfo.csv"
        header = not os.path.isfile(pat_info_fn)
        pat_info_df.to_csv(pat_info_fn, index=False, mode='a',header=header)
        pass


    def run(self, file_extension:str='.lay', mtg_t:str='ir', force_recalc:bool=False)->None:
        """
        This function orchestrates the entire wdw analysis process.

        It performs the following steps:
        1. Retrieves all files with the specified extension (default is '.lay'), from the specified directory.
        2. Calculates the total duration of each sleep stage for the patient.
        3. Cumulates for each slep stage, the signals from the windows that temporally coincide with spike events from any channel
        using a matrix where each row corresponds to an EEG channel.

        Parameters:
        file_extension (str): The file extension to filter for. Default is '.lay'.
        mtg_t (str): The montage type to use for EEG data reading. Default is 'ir'. Options:
            'sr' = Scalp Referential
            'sb' = Scalp Bipolar
            'ir' = Intracranial Referential
            'ib' = Intracranial Bipolar
        force_recalc (bool): A flag indicating whether force_recalc.

        Returns:
        None
        """
        self.get_files_in_folder(file_extension)

        assert len(self.pat_files_ls) > 0, f"No EEG files found in folder {self.ieeg_data_path}"
        #assert len(self.pat_files_ls) >= 40, f"Duration of EEG data is less than 48 hours for patient {self.pat_id}"

        self.pat_files_ls = np.sort(self.pat_files_ls)

        rec_start_idx = 0
        rec_end_idx = 52#48
        if len(self.pat_files_ls) < rec_end_idx:
            ##rec_start_idx = 0
            rec_end_idx = len(self.pat_files_ls)+1
        self.pat_files_ls = self.pat_files_ls[rec_start_idx:rec_end_idx]
        #self.pat_files_ls = self.pat_files_ls[0:52]

        
        sleep_stage_secs_counter_dict = self.get_sleep_stages_duration_sec()
        for k,v in sleep_stage_secs_counter_dict.items():
            if k not in ["Unknown", "NaN"]:
                stage_cum_duration_min = np.round(sleep_stage_secs_counter_dict[k]/60, decimals=2)
                try:
                    assert (stage_cum_duration_min>=0.05), f"Sleep stage {k} duration is {stage_cum_duration_min}, less than 30 min. for patient {self.pat_id}"
                except AssertionError as e:
                    print(f"{e}")
                    return     

        ni_th = 1
        parral_jobs_nr = -1
        eeg_reader = EEG_IO(eeg_filepath=self.pat_files_ls[0], mtg_t=mtg_t)
        eeg_fs = int(eeg_reader.fs)

        if eeg_fs < 1000:
            parral_jobs_nr = -1
        elif eeg_fs < 2000:
            parral_jobs_nr = 8
        else:
            parral_jobs_nr = 2

        try:
            print(f"Processing {self.pat_id} with {parral_jobs_nr} parallel jobs...")
            Parallel(n_jobs=parral_jobs_nr)(delayed(self.get_channel_avg_wdw_vectorized)(this_eeg_fpath, mtg_t, force_recalc, ni_th=ni_th) for this_eeg_fpath in self.pat_files_ls)
        except Exception as e:
            print(f"Error during parallel processing: {e}")
            try:
                print(f"Retrying with fallback processing... {self.pat_id}")
                parral_jobs_nr = 1
                print(f"Processing {self.pat_id} with {parral_jobs_nr} parallel jobs...")
                Parallel(n_jobs=parral_jobs_nr)(delayed(self.get_channel_avg_wdw_vectorized)(this_eeg_fpath, mtg_t, force_recalc, ni_th=ni_th) for this_eeg_fpath in self.pat_files_ls)
            except Exception as e:
                print(f"Error during fallback processing: {e}")
                return

        try:
            #force_recalc = True
            self.get_spike_occ_rate_by_sleep_stage(file_extension=file_extension, mtg_t=mtg_t, force_recalc=force_recalc)
            self.get_overall_ch_stage_spike_amplitude(file_extension=file_extension, mtg_t=mtg_t)
        except Exception as e:
            print(f"Error during final processing: {e}")
            return

        return

    def analyze_sleep(self, file_extension):
        self.get_files_in_folder(file_extension)

        assert len(self.pat_files_ls) > 0, f"No EEG files found in folder {self.ieeg_data_path}"
        #assert len(self.pat_files_ls) >= 40, f"Duration of EEG data is less than 48 hours for patient {self.pat_id}"

        self.pat_files_ls = np.sort(self.pat_files_ls)

        sleep_stage_secs_counter_dict = self.get_sleep_stages_duration_sec()

        sleep_dict = {'PatID':[self.pat_id]*len(sleep_stage_secs_counter_dict.keys()),  
                      'Stage':list(sleep_stage_secs_counter_dict.keys()), 
                      'StageDurationH': [v/3600 for v in sleep_stage_secs_counter_dict.values()]}

        return pd.DataFrame(sleep_dict)

 

    def get_spike_occ_rate_by_sleep_stage(self, file_extension:str='.lay', mtg_t:str='ir', force_recalc:bool=False)->None:
        
        stage_spike_colect_dict = {'Stage':[], 'StageDurationS':[], 'NrSpikes':[]}
        
        pat_stage_spike_occrate_fn = self.output_path / "Stage_Spike_Occurrence_Rate" / f"{self.pat_id}_StageSpikeOccurrenceRate.csv"
        os.makedirs(pat_stage_spike_occrate_fn.parents[0], exist_ok=True)

        if not os.path.isfile(pat_stage_spike_occrate_fn) or force_recalc:

            for this_eeg_fpath in self.pat_files_ls:
                eeg_reader = EEG_IO(eeg_filepath=this_eeg_fpath, mtg_t=mtg_t)
                spike_wdw_indices, spk_df = self.get_detailed_spike_event(this_eeg_fpath, eeg_reader)

                sleep_data_df = self.read_sleep_stages_data(this_eeg_fpath)
                spike_sleep_stage_code = sleep_data_df.I1_1.to_numpy()
                spike_sleep_stage_code = spike_sleep_stage_code[np.logical_not(np.isnan(spike_sleep_stage_code))]
                spike_sleep_stage_name = np.array([self.sleep_stages_map[int(ss_code)] for ss_code in spike_sleep_stage_code])
                for sname in np.unique(spike_sleep_stage_name):
                    stage_name = str(sname)
                    print(stage_name)
                    stage_spike_colect_dict['Stage'].append(stage_name)
                    stage_spike_colect_dict['StageDurationS'].append(np.sum(spike_sleep_stage_name==stage_name))
                    stage_spike_colect_dict['NrSpikes'].append(np.sum(spk_df.stage_name==stage_name))
                    pass
                pass

            stage_spike_colect_df = pd.DataFrame(stage_spike_colect_dict)
            stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
            pat_stage_spike_occrate_dict = {'PatID':[], 'Stage':[], 'StageDurM':[], 'SpikeOccRate':[]}
            for stage_name in stages_ls:
                stage_spike_cnt = np.sum(np.array(stage_spike_colect_dict['NrSpikes'])[np.array(stage_spike_colect_df.Stage)==stage_name])
                stage_dur_s = np.sum(np.array(stage_spike_colect_dict['StageDurationS'])[np.array(stage_spike_colect_df.Stage)==stage_name])
                stage_dur_min = stage_dur_s/60
                #stage_spike_occrate = stage_spike_cnt/stage_dur_min
                #stage_spike_occrate = (stage_spike_cnt)/((len(eeg_reader.ch_names)*stage_dur_s)/60)
                stage_spike_occrate = stage_spike_cnt/len(eeg_reader.ch_names)/stage_dur_min
                pat_stage_spike_occrate_dict['PatID'].append(self.pat_id)
                pat_stage_spike_occrate_dict['Stage'].append(stage_name)
                pat_stage_spike_occrate_dict['StageDurM'].append(stage_dur_min)
                pat_stage_spike_occrate_dict['SpikeOccRate'].append(stage_spike_occrate)
                pass

            pat_stage_spike_occrate_df = pd.DataFrame(pat_stage_spike_occrate_dict)
            pat_stage_spike_occrate_df.to_csv(pat_stage_spike_occrate_fn, index=False)
            pass

    def get_overall_ch_stage_spike_amplitude(self, file_extension:str='.lay', mtg_t:str='ir'):
        spike_data_colect_dict = {'Channel':[], 'Stage':[], 'Amplitude':[], 'NrSpikes':[]}
        for this_eeg_fpath in self.pat_files_ls:
            eeg_reader = EEG_IO(eeg_filepath=this_eeg_fpath, mtg_t=mtg_t)
            spike_cumulator_fn = self.output_path / f"CumulatedSpikes/{eeg_reader.filename.replace(".dat", '_AvgWdwCumulator.pickle')}"
            if os.path.isfile(spike_cumulator_fn):
                print(this_eeg_fpath.name)
                spk_cum = self.load_spike_cumulator(spike_cumulator_fn)
                sleep_stages_ls = list(self.sleep_stages_map.values())

                for sleep_stage in sleep_stages_ls:
                    if sleep_stage != "Unknown":
                        for ch_idx, chname in enumerate(spk_cum.eeg_channels_ls):
                            nr_spikes = spk_cum.nr_spikes[sleep_stage][ch_idx][0]
                            avg_spike_signal = spk_cum.spike_cum_dict[sleep_stage][ch_idx]
                            amplitude = np.max(avg_spike_signal)-np.min(avg_spike_signal)
                            if np.isnan(amplitude):
                                pass
                            if chname=='c105':
                                pass
                            is_natus_virtual_channel = self.is_natus_virtual_channel(mtg_name=chname)
                            if not is_natus_virtual_channel:
                                # if chname=='tples1' and sleep_stage=='N1':
                                #     pass

                                if nr_spikes==0:
                                    amplitude = 0

                                spike_data_colect_dict['Channel'].append(chname)
                                spike_data_colect_dict['Stage'].append(sleep_stage)
                                spike_data_colect_dict['Amplitude'].append(amplitude)
                                spike_data_colect_dict['NrSpikes'].append(nr_spikes)
                                #    pass
                            else:
                                pass


            else:
                print(f"File {spike_cumulator_fn} not found")

            pass
        pass

        # SOZ
        pat_id = self.pat_id
        szr_info_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_clinicalSzrInfo.csv'
        szr_info_fpath = self.szr_info_data_path/ szr_info_fn
        if "Pediatric" in str(szr_info_fpath):
            soz_df = pd.read_csv(szr_info_fpath)
            soz_chann_ls = soz_df['Label'].to_list()
            soz_chann_ls = [c.lower().strip().split('-')[0] for c in soz_chann_ls if isinstance(c,str)]
        else:
            soz_chann_ls = self.parse_szr_info_file(szr_info_fpath)
        soz_chann_ls = [c.lower() for c in soz_chann_ls]

        if len(soz_chann_ls)==0:
            pass

        spike_data_colect_df = pd.DataFrame(spike_data_colect_dict)
        spike_channels_ls = np.sort(spike_data_colect_df['Channel'].unique())
        spike_charact_dict = {'Stage':[], 'Channel':[], 'Amplitude':[], 'NrClipsWithSpikes':[], 'SOZ':[]}

        soz_hits = [spike_data_colect_df.Channel.str.fullmatch(soz_chname.lower(), case=False).sum() for soz_chname in spike_channels_ls]
        #assert np.unique(soz_hits).size==1, f"Not all processing cycles assigned values to a SOZ channel for {self.pat_id}"
        assert 0 not in np.unique(soz_hits), f"SOZ channels not found in EEG for {self.pat_id}"

        for sleep_stage in sleep_stages_ls:
            if sleep_stage != "Unknown":
                for ch_idx, chname in enumerate(spike_channels_ls):
                    spk_ampl_vec = spike_data_colect_df.Amplitude[np.logical_and((spike_data_colect_df['Stage']==sleep_stage).to_numpy(), (spike_data_colect_df['Channel']==chname).to_numpy())]
                    spk_ampl_vec = spk_ampl_vec[spk_ampl_vec>0]
                    if len(spk_ampl_vec)==0:
                        channel_spike_ampl = 0
                    else:
                        channel_spike_ampl = np.median(spk_ampl_vec)
                        #channel_spike_ampl = np.mean(spk_ampl_vec)
                        if np.isnan(channel_spike_ampl):
                            channel_spike_ampl = 0
                        
                    spike_charact_dict['Stage'].append(sleep_stage)
                    spike_charact_dict['Channel'].append(chname)
                    spike_charact_dict['Amplitude'].append(channel_spike_ampl)
                    spike_charact_dict['NrClipsWithSpikes'].append(len(spk_ampl_vec))

                    soz_flag = 0
                    if chname in soz_chann_ls:
                        soz_flag = 1
                    spike_charact_dict['SOZ'].append(soz_flag)
                    pass
            pass
        pass
        
        if np.sum(spike_charact_dict['SOZ']) == 0:
            pass

        spike_charact_df = pd.DataFrame(spike_charact_dict)
        spike_charact_fn = self.output_path / "Spike_Characterized_Channels" / f"{pat_id}_AvgSpikeWdwActivity.csv"
        os.makedirs(spike_charact_fn.parents[0], exist_ok=True)
        spike_charact_df.to_csv(spike_charact_fn, index=False)
        pass



   
    def get_detailed_spike_event(self, eeg_fpath, eeg_reader):
        """
        Extracts detailed spike event information from EEG data.

        Parameters:
        - eeg_fpath (str or Path): The file path to the EEG data file.
        - eeg_reader (object): An object that provides methods to read EEG data, including the sampling frequency (fs) and the number of samples (n_samples).

        Returns:
        - sleep_data_df (DataFrame): A DataFrame containing sleep stage data.
        - spike_data_df (DataFrame): A DataFrame containing spike event data, sorted by time.

        The method performs the following steps:
        1. Reads sleep stage data and spike detection data from the EEG file.
        2. Sorts the spike data by time.
        3. If no spikes are detected, returns None for both DataFrames.
        4. Calculates the indices of the spike windows based on the spike center times and the sampling frequency.
        5. Deletes invalid spike indices (e.g., spikes that start before the beginning of the recording or end after the end of the recording).
        6. Returns the filtered spike data and sleep stage data.

        Example usage:
        sleep_data_df, spike_data_df = get_detailed_spike_event(eeg_fpath, eeg_reader)
        """
        # Read sleep data and spike detections
        sleep_data_df = self.read_sleep_stages_data(eeg_fpath)
        spike_data_df = self.read_spike_data(eeg_fpath).sort_values(by=['Time'], ascending=True)

        if len(spike_data_df)==0:
            return None, None

        #  Get the indices of the spike windows
        spk_wdw_dur_s = 1
        spikes_polarity_vec = spike_data_df['Sign'].to_numpy()
        spikes_center_sec_vec = spike_data_df['Time'].to_numpy()
        spikes_center_samples = (spikes_center_sec_vec*eeg_reader.fs).astype(int)
        spikes_start_samples = ((spikes_center_sec_vec-(spk_wdw_dur_s/2))*eeg_reader.fs).astype(int)
        spikes_end_samples = (spikes_start_samples + (spk_wdw_dur_s*eeg_reader.fs)).astype(int)
        
        # Delete invalid spike indices
        spikes_to_keep = np.logical_not(np.logical_or(spikes_start_samples<0, spikes_end_samples>=eeg_reader.n_samples))
        spikes_polarity_vec = spikes_polarity_vec[spikes_to_keep]
        spikes_center_sec_vec = spikes_center_sec_vec[spikes_to_keep]
        spikes_center_samples = spikes_center_samples[spikes_to_keep]
        spikes_start_samples = spikes_start_samples[spikes_to_keep]
        spikes_end_samples = spikes_end_samples[spikes_to_keep]

        # Get the sleep stage of each spike
        spike_sleep_stage_code = np.array([sleep_data_df.I1_1[sleep_data_df.Time==int(np.round(sc_sec))].to_numpy()[0] for sc_sec in spikes_center_sec_vec])
        spikes_to_keep = np.logical_not(np.isnan(spike_sleep_stage_code))
        spike_sleep_stage_code = spike_sleep_stage_code[spikes_to_keep]
        spikes_polarity_vec = spikes_polarity_vec[spikes_to_keep]
        spikes_center_sec_vec = spikes_center_sec_vec[spikes_to_keep]
        spikes_center_samples = spikes_center_samples[spikes_to_keep]
        spikes_start_samples = spikes_start_samples[spikes_to_keep]
        spikes_end_samples = spikes_end_samples[spikes_to_keep]
        spike_sleep_stage_name = np.array([self.sleep_stages_map[int(ss_code)] for ss_code in spike_sleep_stage_code])

        if np.sum(spikes_to_keep==False)==len(spikes_to_keep):
            return None, None

        # Get the NI from each spike
        #spike_wdw_ni = self.get_spike_wdw_NI(eeg_fpath, eeg_reader, spikes_center_samples)
        spike_wdw_ni= np.zeros_like(spikes_end_samples)
        # Create list of ranges
        ranges = [(start, end) for start, end in zip(spikes_start_samples, spikes_end_samples)]
        # Create list of indices
        spike_wdw_indices = np.r_[tuple(slice(start, end) for start, end in ranges)]

        spikes_info_df = pd.DataFrame({
                            'stage_code':spike_sleep_stage_code,
                            'stage_name':spike_sleep_stage_name, 
                            'polarity':spikes_polarity_vec, 
                            'center_sec':spikes_center_sec_vec, 
                            'center_sample':spikes_center_samples, 
                            'start_sample':spikes_start_samples, 
                            'end_sample':spikes_end_samples,
                            'spike_wdw_ni':spike_wdw_ni}
                            )

        return spike_wdw_indices, spikes_info_df
    
    def get_spike_wdw_NI(self, eeg_fpath:str="", eeg_reader:EEG_IO=None, spikes_center_samples:list[int]=0):
        all_ch_sigs = eeg_reader.get_data()
        spikes_center_time = (spikes_center_samples/eeg_reader.fs)
        ni_tvec, chspec_ni_vec, chavg_ni_vec = get_noise_index_vec(fs=eeg_reader.fs, mtg_labels=eeg_reader.ch_names, mtg_signals=all_ch_sigs, notched=False, is_ieeg=True)
        spikes_wdw_ni_idx = np.floor(spikes_center_time/10).astype(int)
        spike_wdw_ni_vec = chavg_ni_vec[spikes_wdw_ni_idx]
        return spike_wdw_ni_vec
    
    def get_channel_avg_wdw_vectorized(self, this_pat_eeg_fpath, mtg_t:str='ir', force_recalc:bool=False, ni_th:float=1.0)->None:
        """
        This function cumulates for each channel and sleep stage, the windows that temporally coincide with a spike event coming from any channel.

        Parameters:
        mtg_t (str): The montage type to use for EEG data reading. Default is intracranial referential='ir'.
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """
        eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t=mtg_t)
        sig_wdw_fs = eeg_reader.fs

        print(self.output_path.stem)
        print(f"Sampling frequency: {eeg_reader.fs}")

        spike_cumulator_fn = self.output_path / f"CumulatedSpikes/{eeg_reader.filename.replace(".dat", '_AvgWdwCumulator.pickle')}"
        if not os.path.isfile(spike_cumulator_fn) or force_recalc:
            print(this_pat_eeg_fpath.name)
            sleep_stages_ls = list(self.sleep_stages_map.values())
            self.spike_cumulator = AvgWdwCumulator(eeg_channels_ls=eeg_reader.ch_names, sleep_stage_ls=sleep_stages_ls, sig_wdw_dur_s=1, sig_wdw_fs=sig_wdw_fs)
            fs = eeg_reader.fs
            fs_us = self.spike_cumulator.get_undersampling_frequency()

            spike_wdw_indices, spk_df = self.get_detailed_spike_event(this_pat_eeg_fpath, eeg_reader)
            if spk_df is None:
                self.save_spike_cumulator(filepath=spike_cumulator_fn)
                return
            
            start_indices = spk_df.start_sample.to_numpy()
            end_indices = spk_df.end_sample.to_numpy()
            polarity_vec = spk_df.polarity.to_numpy()
            nr_total_spikes = len(spk_df.center_sample)
            if nr_total_spikes == 0:
                self.save_spike_cumulator(filepath=spike_cumulator_fn)
                return

            assert len(start_indices)==len(end_indices), f"Start and end indices do not match for {this_pat_eeg_fpath.name}"

            ###
            # 1. Read the EEG one hour at a time and extract the spike windows
            # This is done to avoid memory issues when processing large EEG files
            spike_wdws = np.zeros((len(eeg_reader.ch_names), nr_total_spikes, fs_us))
            for hour_start in range(0, eeg_reader.n_samples, eeg_reader.fs * 3600):
                hour_stop = min(hour_start + eeg_reader.fs * 3600, eeg_reader.n_samples)
                all_ch_sigs = eeg_reader.get_data(start=hour_start, stop=hour_stop)
                for spike_idx, spike_locs in enumerate(zip(start_indices, end_indices)):
                    if spike_locs[0] < hour_start or spike_locs[1] > hour_stop:
                        continue
                    try:
                        spike_wdws[:, spike_idx, :] = all_ch_sigs[:, spike_locs[0]-hour_start:spike_locs[1]-hour_start]
                    except ValueError as e:
                        print(f"Error processing spike window for {this_pat_eeg_fpath.name} at index {spike_idx}: {e}")
                        continue

                # Show progress
                try:
                    if hour_start % int(eeg_reader.fs * 3600) == 0:
                        print(f"{this_pat_eeg_fpath.name} = {hour_start / eeg_reader.n_samples * 100:.2f}%")
                except ZeroDivisionError:
                    pass
            ###

            # ###
            # # 2. Read the EEG data containing spikes and extract the spike windows
            # #all_ch_sigs = eeg_reader.get_data()
            # spike_wdws = np.zeros((len(eeg_reader.ch_names), nr_total_spikes, fs_us))
            # for spike_idx, spike_locs in enumerate(zip(start_indices, end_indices)):
            #     all_channs_spike_signal = eeg_reader.get_data(start=spike_locs[0], stop=spike_locs[1])

            #     # under sample the spike signal to the desired frequency
            #     for eeg_chi, ch_name in enumerate(eeg_reader.ch_names):
            #         # Read the EEG segment containing a spike and undersample it
            #         #spike_wdws[eeg_chi, spike_idx] = self.undersample_signal(all_channs_spike_signal[eeg_chi], fs_us)
            #         spike_wdws[eeg_chi, spike_idx] = all_channs_spike_signal[eeg_chi]
            #         assert not np.any(spike_wdws[eeg_chi, spike_idx] - all_channs_spike_signal[eeg_chi])
            #         pass
                
            #     try:
            #         if spike_idx%int(nr_total_spikes/100) == 0:
            #             print(f"{this_pat_eeg_fpath.name} = {(spike_idx+1)/nr_total_spikes*100:.2f}%")
            #     except ZeroDivisionError:
            #         pass
            # ###
                    
            for eeg_chi, ch_name in enumerate(eeg_reader.ch_names):
                for k, stage_name in self.sleep_stages_map.items():
                    spike_wdw_sel_indices = np.where(spk_df.stage_name==stage_name)
                    if len(spike_wdw_sel_indices[0])>0:
                        stage_spikes_polarity = polarity_vec[spike_wdw_sel_indices]
                        stage_spike_wdws = spike_wdws[eeg_chi][spike_wdw_sel_indices]
                        
                        # Correct spike signals polarity
                        stage_spike_wdws = stage_spike_wdws * stage_spikes_polarity.reshape(-1, 1) # correct spike signals polarity
                        
                        # Remove DC offset from the spike signals
                        for spk_sig_idx, spike_sig in enumerate(stage_spike_wdws):
                            dc_offset = np.median(spike_sig)
                            stage_spike_wdws[spk_sig_idx] = spike_sig - dc_offset
                            pass

                        # #Get absolute value of the spike signals
                        #stage_spike_wdws = np.abs(stage_spike_wdws)

                        avg_spike = np.mean(stage_spike_wdws, axis=0)
                        nr_spikes_in_avg = stage_spike_wdws.shape[0]
                        self.spike_cumulator.add_spike(sleep_stage=stage_name, ch_name=ch_name, avg_spike_signal=avg_spike, nr_spikes=nr_spikes_in_avg)
                        pass

                pass
            
            self.save_spike_cumulator(filepath=spike_cumulator_fn)
        else:
            print(f"Spike cumulator file {spike_cumulator_fn} already exists. Skipping calculation.")
        pass

    def get_unique_channels_from_eegs(self, eeg_files_ls):
        """
        This function retrieves all unique channels from the EEG files.

        Parameters:
        None

        Returns:
        None
        """
        ch_names = []
        for eeg_fpath in eeg_files_ls:
            eeg_reader = EEG_IO(eeg_fpath)
            ch_names.extend(eeg_reader.ch_names)
            pass
        unique_channels = list(set(ch_names))
        return unique_channels
        
    def get_nr_spikes(self, eeg_files_ls):
        sleep_stages_ls=list(self.sleep_stages_map.values())
        spk_cntr = {stage:[0] for stage in sleep_stages_ls}
        for eeg_fpath in eeg_files_ls:
            spike_cumulator_fpath = self.output_path / f"CumulatedSpikes/{eeg_fpath.name.replace(".lay", '_AvgWdwCumulator.pickle')}"
            spk_cum = self.load_spike_cumulator(spike_cumulator_fpath)
            for sleep_stage in sleep_stages_ls:
                spk_cntr[sleep_stage][0] += spk_cum.spike_counter[sleep_stage][0]
        for sleep_stage in sleep_stages_ls:
            if spk_cntr[sleep_stage][0] == 0:
                print(f"No spikes found for stage {sleep_stage}")
                pass
        return spk_cntr
    
    def get_sleep_stages_duration(self, eeg_files_ls):
        stages_duration = {stage:0 for stage in self.sleep_stages_map.values()}
        for eeg_fpath in eeg_files_ls:
            sleep_data_df = self.read_sleep_stages_data(eeg_fpath)
            for stage_code in self.sleep_stages_map.keys():
                stages_duration[self.sleep_stages_map[stage_code]] += np.sum(sleep_data_df.I1_1==stage_code)
                pass
            pass
        return stages_duration
    
    def get_spikes_avg_amp(self, eeg_files_ls, chanels_ls):
        spk_cntr = self.get_nr_spikes(eeg_files_ls)
        spk_amplitude = {stage:{chname:0 for chname in chanels_ls} for stage in self.sleep_stages_map.values()}
        for eeg_fpath in eeg_files_ls:
            spike_cumulator_fpath = self.output_path / f"CumulatedSpikes/{eeg_fpath.name.replace(".lay", '_AvgWdwCumulator.pickle')}"
            spk_cum = self.load_spike_cumulator(spike_cumulator_fpath)
            for stage in self.sleep_stages_map.values():
                # if spk_cntr[stage][0]==0:
                #     continue
                for chname in chanels_ls:
                    cum_avg_spike = spk_cum.spike_cum_dict[stage][spk_cum.get_ch_idx(chname)]
                    cum_avg_spike_ampl = np.max(cum_avg_spike)-np.min(cum_avg_spike)
                    nr_spikes_in_avg = spk_cum.spike_counter[stage][0]
                    # spk_amplitude[stage][chname] += (nr_spikes_in_avg/spk_cntr[stage][0])*cum_avg_spike_ampl
                    spk_amplitude[stage][chname] += (nr_spikes_in_avg*cum_avg_spike_ampl)

                    pass
            pass
        for stage in self.sleep_stages_map.values():
            tot_stage_spike_cnt = spk_cntr[stage][0]
            for chname in chanels_ls:
                if tot_stage_spike_cnt==0:
                    print(f"No spikes found for stage {stage}")
                else:
                    spk_amplitude[stage][chname] = spk_amplitude[stage][chname]/tot_stage_spike_cnt
        return spk_amplitude


    def get_avg_wdw_by_day_by_ch(self, eeg_filepaths_byday_df, mtg_t, force_recalc):

        eeg_files_ls = eeg_filepaths_byday_df.EEG_Filepath.values.tolist()
        chanels_ls = self.get_unique_channels_from_eegs(eeg_files_ls)        
        sleep_stages_ls = list(self.sleep_stages_map.values())

        # Get the sampling rate used to cumulate the spike windows
        spike_cum_fpath_temp = self.output_path / f"CumulatedSpikes/{eeg_filepaths_byday_df.EEG_Filepath[0].name.replace(".lay", '_AvgWdwCumulator.pickle')}"
        fs = self.load_spike_cumulator(spike_cum_fpath_temp).sig_wdw_fs

        avg_spike_by_day_stage_ch_df = pd.DataFrame()
        for day_nr in eeg_filepaths_byday_df.EEG_Day.unique().tolist():
            day_eeg_files_ls = eeg_filepaths_byday_df.EEG_Filepath[eeg_filepaths_byday_df.EEG_Day==day_nr].values.tolist()
            spk_cntr = self.get_nr_spikes(day_eeg_files_ls)
            sleep_stages_durs = self.get_sleep_stages_duration(day_eeg_files_ls)
            day_spk_amplitude = self.get_spikes_avg_amp(day_eeg_files_ls, chanels_ls)
            nr_chs = len(chanels_ls)
            for stage in sleep_stages_ls:
                data_df = {'Stage':[stage]*nr_chs, 
                            'DayNr':[day_nr]*nr_chs, 
                            'NrHourRecords':[len(day_eeg_files_ls)]*nr_chs, 
                            'NrSpikeWdws':[spk_cntr[stage][0]]*nr_chs, 
                            'ChName':chanels_ls, 
                            'AvgSpikeAmplitude':[]}
                
                for ch_name in chanels_ls:
                    data_df['AvgSpikeAmplitude'].append(day_spk_amplitude[stage][ch_name])
                    pass
                data_df = pd.DataFrame(data_df)
                avg_spike_by_day_stage_ch_df = pd.concat([avg_spike_by_day_stage_ch_df, data_df], ignore_index=True)
                pass
        return avg_spike_by_day_stage_ch_df
                

    def get_channel_coordinates(self, avg_spike_by_day_stage_ch_df):
        spike_dets_chnames = avg_spike_by_day_stage_ch_df.ChName.unique().tolist()
        # Load channel coordinates
        pat_id = self.pat_id
        ch_coords_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_elecInfo.csv'
        coords_fpath = self.ch_coordinates_data_path/ ch_coords_fn
        ch_coords_data = pd.read_csv(coords_fpath)[['name', 'x', 'y', 'z']]

        # Check if all channels used for the spike detection can be given 3D localization coordinates
        assert np.sum([c.lower() in ch_coords_data.name.str.lower().to_list() for c in spike_dets_chnames]) == len(spike_dets_chnames)

        ch_coords_dict = defaultdict(set)#{chname.lower():(0,0,0) for chname in spike_dets_chnames}
        for chname in spike_dets_chnames:
            sel_ch_coords_data = ch_coords_data[ch_coords_data.name.str.fullmatch(chname.lower(), case=False)].reset_index(drop=True)

            # Check that only one set of coordinateds is being selected for the specific channel
            assert len(sel_ch_coords_data)==1

            x_coord=0
            y_coord=0
            z_coord=0
            try:
                x_coord = float(sel_ch_coords_data.x[0])
                y_coord = float(sel_ch_coords_data.y[0])
                z_coord = float(sel_ch_coords_data.z[0])

            except:
                print(f"Invalid coordinates in channel: {chname}")
                continue
            coords_are_nonzero = x_coord>0 and y_coord>0 and z_coord>0
            ch_coords_dict[sel_ch_coords_data.name[0].lower()] = [x_coord, y_coord, z_coord]

        avg_spike_by_day_stage_ch_df['x'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        avg_spike_by_day_stage_ch_df['y'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        avg_spike_by_day_stage_ch_df['z'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        #avg_spike_by_day_stage_ch_df['xyz'] = avg_spike_by_day_stage_ch_df.ChName.str.lower().map(lambda x[0]: ch_coords_dict[x])
        for chname in ch_coords_dict.keys():
            sel_rows = avg_spike_by_day_stage_ch_df.ChName.str.fullmatch(chname, case=False)
            if len(sel_rows)>0:
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'x'] = ch_coords_dict[chname][0]
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'y'] = ch_coords_dict[chname][1]
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'z'] = ch_coords_dict[chname][2]
                pass

        del_rows = avg_spike_by_day_stage_ch_df[avg_spike_by_day_stage_ch_df.x==-1].index
        avg_spike_by_day_stage_ch_df.drop(del_rows, inplace=True)
        #avg_spike_by_day_stage_ch_df = avg_spike_by_day_stage_ch_df[avg_spike_by_day_stage_ch_df.xyz.apply(len)>0]

        return avg_spike_by_day_stage_ch_df
    

    def parse_szr_info_file(self, szr_info_fpath):
        szr_info_df = pd.read_csv(szr_info_fpath)
        szr_info_df = szr_info_df[['vig.', 'origin']].reset_index()
        ch_szr_involvment_dict = {'Origin': [], 'Early': [], 'Late': []}
        for i in np.arange(len(szr_info_df)):
            szr_info_str = szr_info_df.at[i, 'origin']
            szr_info_str = szr_info_str.replace(' ', '')

            origin_idx = szr_info_str.lower().find('origin')
            early_idx = szr_info_str.lower().find('early')
            late_idx = szr_info_str.lower().find('late')
            origin_channs_start = origin_idx+7
            origin_channs_end = early_idx

            early_channs_start = early_idx+6
            early_channs_end = late_idx

            late_channs_start = late_idx+5
            late_channs_end = len(szr_info_str)

            origin_channs = szr_info_str[origin_channs_start:origin_channs_end]
            early_channs = szr_info_str[early_channs_start:early_channs_end]
            late_channs = szr_info_str[late_channs_start:late_channs_end]

            origin_channs_ls = origin_channs.split(",")
            early_channs_ls = early_channs.split(",")
            late_channs_ls = late_channs.split(",")

            if len(origin_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Origin'].extend(origin_channs_ls)
            if len(early_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Early'].extend(early_channs_ls)
            if len(late_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Late'].extend(late_channs_ls)

        pass

        soz_chann_ls = list(set(ch_szr_involvment_dict['Origin']))
        # soz_chann_ls.extend(list(set(ch_szr_involvment_dict['Early'])))
        # soz_chann_ls = list(set(soz_chann_ls))
        soz_chann_ls = [c.lower() for c in soz_chann_ls]
        assert len(soz_chann_ls)>0, "No SOZ channels found in the seizure info file"

        if '1096' in szr_info_fpath.name:
            soz_chann_ls = correct_relabelled_chnames(soz_chann_ls, 1096)
        elif '273' in szr_info_fpath.name:
            soz_chann_ls = correct_relabelled_chnames(soz_chann_ls, 273)
        elif '264' in szr_info_fpath.name:
            soz_chann_ls = correct_relabelled_chnames(soz_chann_ls, 264)
        
        soz_chann_ls = EEG_IO.clean_channel_labels(None, soz_chann_ls)

        return soz_chann_ls

    # def get_soz_info(self, avg_spike_by_day_stage_ch_df):
    #     # Load channel coordinates
    #     pat_id = self.pat_id
    #     szr_info_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_clinicalSzrInfo.csv'
    #     szr_info_fpath = self.szr_info_data_path/ szr_info_fn
    #     soz_chann_ls = self.parse_szr_info_file(szr_info_fpath)

    #     soz_chann_ls = [c.lower() for c in soz_chann_ls]
    #     avg_spike_by_day_stage_ch_df['SOZ'] = avg_spike_by_day_stage_ch_df.ChName.str.lower().map(lambda x: x in soz_chann_ls)

    #     assert len(avg_spike_by_day_stage_ch_df['SOZ'].unique())>1, "No channels could be assigned to a SOZ"

    #     return avg_spike_by_day_stage_ch_df

    
    def get_weighted_avg_coordinates_deprecated(self, stage_df):

        chavg_spike_chname = stage_df.ChName.to_numpy()
        chavg_spike_ampl = self.MinMaxScaler(stage_df.AvgSpikeAmplitude.to_numpy())
        x_coords = stage_df.x.to_numpy()
        y_coords = stage_df.y.to_numpy()
        z_coords = stage_df.z.to_numpy()
        
        max_ampl_x = x_coords[np.argmax(chavg_spike_ampl)]
        max_ampl_y = y_coords[np.argmax(chavg_spike_ampl)]
        max_ampl_z = z_coords[np.argmax(chavg_spike_ampl)]

        coords_weights = (chavg_spike_ampl/np.max(chavg_spike_ampl))
        #coords_weights = np.pow(coords_weights,2)
        #coords_weights = MinMaxScaler().fit_transform(coords_weights.reshape(-1,1)).flatten()

        x_strides = (max_ampl_x-x_coords)*coords_weights
        avg_x_stride = np.mean(x_strides)
        final_x = max_ampl_x-avg_x_stride

        y_strides = (max_ampl_y-y_coords)*coords_weights
        avg_y_stride = np.mean(y_strides)
        final_y = max_ampl_y-avg_y_stride

        z_strides = (max_ampl_z-z_coords)*coords_weights
        avg_z_stride = np.mean(z_strides)
        final_z = max_ampl_z-avg_z_stride

        weighted_x = [int(final_x)]
        weighted_y = [int(final_y)]
        weighted_z = [int(final_z)]

        return weighted_x, weighted_y, weighted_z


    def get_weighted_avg_coordinates(self, stage_df):
        x_coords = stage_df.x.to_numpy()
        y_coords = stage_df.y.to_numpy()
        z_coords = stage_df.z.to_numpy()
        weighted_x = np.mean(x_coords)
        weighted_y = np.mean(y_coords)
        weighted_z = np.mean(z_coords)
        if not (stage_df.AvgSpikeAmplitude == 0).all():
            chavg_spike_ampl = np.power(stage_df.AvgSpikeAmplitude.to_numpy(), np.e)
            weighted_x = np.round(np.sum(chavg_spike_ampl * x_coords) / np.sum(chavg_spike_ampl))
            weighted_y = np.round(np.sum(chavg_spike_ampl * y_coords) / np.sum(chavg_spike_ampl))
            weighted_z = np.round(np.sum(chavg_spike_ampl * z_coords) / np.sum(chavg_spike_ampl))

        return [int(weighted_x)], [int(weighted_y)], [int(weighted_z)]


    def plot_daily_spike_activity(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()


        for di, day in enumerate(days_ls):
            # Create a 3D scatter plot for each Sleep Stage    
            fig = make_subplots(
                        rows=1, cols=5,
                        horizontal_spacing = 0.01,  vertical_spacing  = 0.1,
                        subplot_titles=(stages_names_ls),
                        start_cell="top-left",
                        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
                        )
            day_spk_df = spk_df[spk_df.DayNr==day]
            for ss_idx, stage_name in enumerate(stages_names_ls):
                stage_df = day_spk_df[day_spk_df.Stage.str.fullmatch(stage_name, case=False)]

                chavg_spike_chname = stage_df.ChName.to_numpy()
                chavg_spike_ampl = self.MinMaxScaler(stage_df.AvgSpikeAmplitude.to_numpy())
                x_coords = stage_df.x.to_numpy()
                y_coords = stage_df.y.to_numpy()
                z_coords = stage_df.z.to_numpy()

                #Count
                fig.add_trace(
                    go.Scatter3d(
                        x = x_coords,
                        y = y_coords,
                        z = z_coords,
                        mode='markers',  # Show markers and labels
                        marker=dict(
                            size=10,
                            color=chavg_spike_ampl,
                            colorscale='viridis',
                            opacity=0.9,
                            showscale=True,
                            cmin=np.min(chavg_spike_ampl),
                            cmax=np.max(chavg_spike_ampl),
                            colorbar=dict(title="Scaled<br>Amplitude"),#, len=0.50, y=0.8),
                        ),
                        text=[f"{chname}: {amp:.2f}" for chname, amp in zip(chavg_spike_chname, chavg_spike_ampl)],  # Custom text for the fourth dimension
                        hoverinfo='text',
                    ),
                    row=1, col=ss_idx+1
                )

                # Highlight SOZ channels
                soz_x_coords = x_coords[stage_df.SOZ.to_numpy()]
                soz_y_coords = y_coords[stage_df.SOZ.to_numpy()]
                soz_z_coords = z_coords[stage_df.SOZ.to_numpy()]
                fig.add_trace(
                    go.Scatter3d(
                        x = soz_x_coords,
                        y = soz_y_coords,
                        z = soz_z_coords,
                        mode='markers',  # Show markers and labels
                        marker=dict(
                            symbol="circle-open",
                            size=12,
                            color='Cyan',
                            opacity=0.9,
                            showscale=False,
                            line=dict(color='Cyan',width=10)
                        ),
                    ),
                    row=1, col=ss_idx+1
                )

                # Amplitude Weighted Virtual Contact 
                wx, wy, wz = self.get_weighted_avg_coordinates(stage_df)
                fig.add_trace(
                    go.Scatter3d(
                        x = wx,
                        y = wy,
                        z = wz,
                        mode='markers',  # Show markers and labels
                        marker=dict(
                            symbol="diamond-open",
                            size=12,
                            color='Red',
                            opacity=1,
                            showscale=False,
                            line=dict(color='Red',width=10)
                        ),
                    ),
                    row=1, col=ss_idx+1
                )
            

            fig.update_layout(autosize=True)
            #fig.update_layout(autosize=True,width=2048,height=1024)
            fig.update_layout(title_text=f"{pat_id} Day {day}<br>(Avg. Window Amplitude)", showlegend=False)
            # Define the camera settings
            center_dict = {'x': 0, 'y': 0, 'z': 0}
            eye_dict = {'x': 3, 'y': 3, 'z': 3}
            projection_dict = {'type': 'perspective'} # perspective, orthographic
            up_dict = {'x': 0, 'y': 0, 'z': 1}
            camera = dict(center=center_dict, eye=eye_dict, projection=projection_dict, up=up_dict)
            #camera = dict(center=center_dict, eye=eye_dict)
            #camera = dict(center=center_dict, eye=eye_dict,up=up_dict)

            for i in range(len(stages_names_ls)):
                fig.update_layout(**{f'scene{i+1}': dict(camera=camera)})

            fig.show()
            out_images_path = self.output_path /"Images/SpkAmpWAvg_Contact_ByDay"
            os.makedirs(out_images_path, exist_ok=True)
            fig_fpath = out_images_path / f"{pat_id}_Day{day}_Spk_Amp_wAvg_Cntct_Coord.html"    
            fig.write_html(fig_fpath)

            fig.update_layout(autosize=True,width=2048,height=1024)
            fig_fpath = out_images_path / f"{pat_id}_Day{day}_Spk_Amp_wAvg_Cntct_Coord.jpg" 
            fig.write_image(fig_fpath)
            #self.add_synch_subplots_jscript(fig, fig_fpath, stages_names_ls)
            pass
        return fig

    def add_synch_subplots_jscript(self, fig, fig_fpath, stages_names_ls):
        #https://community.plotly.com/t/synchronize-camera-across-3d-subplots/22236/4
        
        # Generate the HTML content with embedded JavaScript
        from plotly.offline import plot

        # Generate the div element containing the figure
        plot_div = plot(fig, output_type='div', include_plotlyjs='cdn', auto_open=False)


        # get the a div
        div = plot(fig, include_plotlyjs=False, output_type='div')
        # retrieve the div id (you probably want to do something smarter here with beautifulsoup)
        div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
        # your custom JS code
        sync_camera_js = '''
            <script>
            var gd = document.getElementById('{div_id}');
            var isUnderRelayout = false

            gd.on('plotly_relayout', () => {{
            console.log('relayout', isUnderRelayout)
            if (!isUnderRelayout) {{
                Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
                .then(() => {{ isUnderRelayout = false }}  )
            }}

            isUnderRelayout = true;
            }})
            </script>'''

        # Combine the plot div and the synchronization script
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Synchronized Scatter3D Subplots</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            {plot_div}
            {sync_camera_js}
        </body>
        </html>
        """

        # Save the combined HTML content to a file
        with open(fig_fpath, 'w') as f:
            f.write(html_content)
        pass



    def get_wavg_coords_by_day(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()
        wavg_coords = {stage:[] for stage in stages_names_ls}
        for di, day in enumerate(days_ls):
            day_spk_df = spk_df[spk_df.DayNr==day]
            for ss_idx, stage_name in enumerate(stages_names_ls):
                stage_df = day_spk_df[day_spk_df.Stage.str.fullmatch(stage_name, case=False)]
                # Amplitude Weighted Virtual Contact 
                wx, wy, wz = self.get_weighted_avg_coordinates(stage_df)
                wavg_coords[stage_name].append([wx[0], wy[0], wz[0]])
                pass
            pass
        pass

        test_nr = 1
        for stg_a in stages_names_ls:
            for stg_b in stages_names_ls:
                res = test_mvmean_2indep(np.array(wavg_coords[stg_a]), np.array(wavg_coords[stg_b]))
                res2 = hotelling_t2(np.array(wavg_coords[stg_a]), np.array(wavg_coords[stg_b]))
                print(f"{test_nr}.Comparing {stg_a} vs {stg_b}, pvalue = {res.pvalue:.5f}")
                print(f"{test_nr}.Comparing {stg_a} vs {stg_b}, pvalue = {res2[2]:.5f}\n")
                test_nr += 1
                pass

        pass

    def get_daily_centroid_shift(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()
        dayily_centroid_shifts = {stage:[] for stage in stages_names_ls}
        dayily_centroid_shifts = {'day_nr':[f"{di-1}->{di}" for di in np.arange(1,len(days_ls))], **dayily_centroid_shifts}
        all_wavg_coords = np.empty((0, 3), dtype=int)
        for ss_idx, stage_name in enumerate(stages_names_ls):
            for di in np.arange(1,len(days_ls)):
                day_a = days_ls[di-1]
                day_b = days_ls[di]
                stage_df_a = spk_df[np.logical_and((spk_df.DayNr==day_a).to_numpy(), (spk_df.Stage.str.fullmatch(stage_name, case=False)).to_numpy())]
                stage_df_b = spk_df[np.logical_and((spk_df.DayNr==day_b).to_numpy(), (spk_df.Stage.str.fullmatch(stage_name, case=False)).to_numpy())]
                # Amplitude Weighted Virtual Contact 
                wx_a, wy_a, wz_a = self.get_weighted_avg_coordinates(stage_df_a)
                wx_b, wy_b, wz_b = self.get_weighted_avg_coordinates(stage_df_b)
                point_a = np.array([wx_a[0], wy_a[0], wz_a[0]])
                point_b = np.array([wx_b[0], wy_b[0], wz_b[0]])
                diff_vec = np.array(point_a) - np.array(point_b)
                dist = np.linalg.norm(diff_vec)
                dist_dlp = np.sqrt(np.sum((np.array(point_b) - np.array(point_a))**2))
                dayily_centroid_shifts[stage_name].append(dist)
                all_wavg_coords = np.vstack((all_wavg_coords, point_a))
                all_wavg_coords = np.vstack((all_wavg_coords, point_b))
                pass
            pass
        pass

        return dayily_centroid_shifts
    
    def plot_daily_centroid_shift(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()
        all_coords = np.empty((0, 3), dtype=int)

        # Create a 3D scatter plot for each Sleep Stage    
        fig = make_subplots(
                    rows=1, cols=5,
                    horizontal_spacing = 0.01,  vertical_spacing  = 0.1,
                    subplot_titles=(stages_names_ls),
                    start_cell="top-left",
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
                    )

        # Get 5 colors from the 'Viridis' color scale
        colors = plotly.colors.sample_colorscale('Viridis', [0, 0.25, 0.5, 0.75, 1.0])
        for ss_idx, stage_name in enumerate(stages_names_ls):
                            
            # Create the figure
            color_val = colors[-1]
            for di in np.arange(1,len(days_ls)):
                day_a = days_ls[di-1]
                day_b = days_ls[di]
                day_spk_df_a = spk_df[spk_df.DayNr==day_a]
                day_spk_df_b = spk_df[spk_df.DayNr==day_b]
                stage_df_a = day_spk_df_a[day_spk_df_a.Stage.str.fullmatch(stage_name, case=False)]
                stage_df_b = day_spk_df_b[day_spk_df_b.Stage.str.fullmatch(stage_name, case=False)]

                # Amplitude Weighted Virtual Contact 
                wx_a, wy_a, wz_a = self.get_weighted_avg_coordinates(stage_df_a)
                wx_b, wy_b, wz_b = self.get_weighted_avg_coordinates(stage_df_b)
                x_diff = wx_b[0] - wx_a[0]
                y_diff = wy_b[0] - wy_a[0]
                z_diff = wz_b[0] - wz_a[0]
                diff_norm = np.linalg.norm(np.array([x_diff,y_diff,z_diff]))
                
                point_a = np.array([wx_a[0], wy_a[0], wz_a[0]])
                point_b = np.array([wx_b[0], wy_b[0], wz_b[0]])
                diff_norm = np.sqrt(np.sum((np.array(point_b) - np.array(point_a))**2))

                print(f"Start: {point_a}\n  End: {point_b}\n Diff: {diff_norm:.2f}")

                # Define the starting points of the vectors
                x_start = np.array(wx_a)  # X-coordinates of starting points
                y_start = np.array(wy_a)  # Y-coordinates of starting points
                z_start = np.array(wz_a)  # Z-coordinates of starting points
                # Define the end points of the vectors
                x_end = np.array(wx_b)  # X-coordinates of starting points
                y_end = np.array(wy_b)  # Y-coordinates of starting points
                z_end = np.array(wz_b)  # Z-coordinates of starting points

                all_coords = np.vstack((all_coords, np.array([x_start[0], y_start[0], z_start[0]])))
                all_coords = np.vstack((all_coords, np.array([x_end[0], y_end[0], z_end[0]])))

                hovertext = f"d{days_ls[di-1]}->d{days_ls[di]}, Start:{point_a}, End:{point_b}, Diff:{diff_norm:.2f}"

                marker_col = color_val
                marker_specs = dict(size=10, color=color_val, opacity=0.7)
                if di == 1:
                    marker_specs = dict(size=[20,10], color=['Green', color_val], opacity=1, line=dict(color=['Green', color_val], width=15))
                elif di == len(days_ls)-1:
                   marker_specs = dict(size=[10,20], color=[color_val, 'Red'], opacity=1, line=dict(color=[color_val, 'Red'], width=15))

                # Add the vectors as lines from the starting points to the end points
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_start[0], x_end[0]],
                        y=[y_start[0], y_end[0]],
                        z=[z_start[0], z_end[0]],
                        mode='lines+markers',
                        line=dict(color='Black', width=6),
                        marker=marker_specs,
                        name=f'Vector {di}',
                        text=hovertext,
                        hoverinfo='text',
                    ),
                    row=1, col=ss_idx+1
                )

        fig.update_layout(autosize=True)
        fig.update_layout(title_text=f"Daily Displacement of Amplitude-Weighted Spike Centroid<br>{pat_id}", showlegend=False)
        # Define the camera settings
        center_dict = {'x': 0, 'y': 0, 'z': 0}
        eye_dict = {'x': 2, 'y': 2, 'z': 2}
        projection_dict = {'type': 'perspective'} # perspective, orthographic
        up_dict = {'x': 0, 'y': 0, 'z': 1}
        camera = dict(center=center_dict, eye=eye_dict, projection=projection_dict, up=up_dict)
        for i in range(len(stages_names_ls)):
            fig.update_layout(**{f'scene{i+1}': dict(camera=camera)})

        alb = 1
        axis_limits = dict(
            xaxis=dict(nticks=10, range=[np.min(all_coords[:,0])-alb, np.max(all_coords[:,0])+alb], title='X Axis'),
            yaxis=dict(nticks=10, range=[np.min(all_coords[:,1])-alb, np.max(all_coords[:,1])+alb], title='Y Axis'),
            zaxis=dict(nticks=10, range=[np.min(all_coords[:,2])-alb, np.max(all_coords[:,2])+alb], title='Z Axis'),
            aspectratio=dict(x=1, y=1, z=1)
        )

        for i in range(len(stages_names_ls)):
            fig.update_layout(**{f'scene{i+1}': axis_limits})
            
        fig.show()
        out_images_path = self.output_path /"Images/Trajectory/"
        os.makedirs(out_images_path, exist_ok=True)
        fig_fpath = out_images_path / f"{pat_id}_AllDays_SpkAmp_wAvgCntct_Trajectory.html"    
        fig.write_html(fig_fpath)

        fig.update_layout(autosize=True,width=2048,height=1024)
        fig_fpath = out_images_path / f"{pat_id}_AllDays_SpkAmp_wAvgCntct_Trajectory.jpg"
        fig.write_image(fig_fpath)
        pass


    def plot_daily_centroid_shift_with_SOZ(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()
        all_coords = np.empty((0, 3), dtype=int)

        # Create a 3D scatter plot for each Sleep Stage    
        fig = make_subplots(
                    rows=1, cols=5,
                    horizontal_spacing = 0.01,  vertical_spacing  = 0.1,
                    subplot_titles=(stages_names_ls),
                    start_cell="top-left",
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
                    )
        
        x_coords = spk_df.loc[np.logical_and(spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False)), 'x'].to_numpy()
        y_coords = spk_df.loc[np.logical_and(spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False)), 'y'].to_numpy()
        z_coords = spk_df.loc[np.logical_and(spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False)), 'z'].to_numpy()

        soz_x_coords = spk_df.loc[np.logical_and.reduce((spk_df.SOZ, spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False))), 'x'].to_numpy()
        soz_y_coords = spk_df.loc[np.logical_and.reduce((spk_df.SOZ, spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False))), 'y'].to_numpy()
        soz_z_coords = spk_df.loc[np.logical_and.reduce((spk_df.SOZ, spk_df.DayNr==1, spk_df.Stage.str.fullmatch('N1', case=False))), 'z'].to_numpy()
        all_coords = np.vstack((all_coords, np.vstack((x_coords, y_coords, z_coords)).transpose()))

        assert len(x_coords)>0, "No coordinates found for the first day"
        assert len(x_coords) == len(spk_df.ChName.unique()), "Coordinates and channels mismatch"

        # Get 5 colors from the 'Viridis' color scale
        colors = plotly.colors.sample_colorscale('Viridis', [0, 0.25, 0.5, 0.75, 1.0])
        for ss_idx, stage_name in enumerate(stages_names_ls):

            # All contacts
            fig.add_trace(
                go.Scatter3d(
                    x = x_coords, y = y_coords, z = z_coords,
                    mode='markers',  # Show markers and labels
                    marker=dict(symbol="circle-open", size=10, color='blue', opacity=0.9, showscale=False, line=dict(color='blue',width=10)),
                ),
                row=1, col=ss_idx+1
            )
            # SOZ contacts
            fig.add_trace(
                go.Scatter3d(
                    x = soz_x_coords, y = soz_y_coords, z = soz_z_coords,
                    mode='markers',  # Show markers and labels
                    marker=dict(symbol="circle", size=10, color='orange', opacity=1, showscale=False, line=dict(color='orange',width=10)),
                ),
                row=1, col=ss_idx+1
            )
                            
            # Create the figure
            color_val = colors[-1]
            for di in np.arange(1,len(days_ls)):
                day_a = days_ls[di-1]
                day_b = days_ls[di]
                day_spk_df_a = spk_df[spk_df.DayNr==day_a]
                day_spk_df_b = spk_df[spk_df.DayNr==day_b]
                stage_df_a = day_spk_df_a[day_spk_df_a.Stage.str.fullmatch(stage_name, case=False)]
                stage_df_b = day_spk_df_b[day_spk_df_b.Stage.str.fullmatch(stage_name, case=False)]

                # Amplitude Weighted Virtual Contact 
                wx_a, wy_a, wz_a = self.get_weighted_avg_coordinates(stage_df_a)
                wx_b, wy_b, wz_b = self.get_weighted_avg_coordinates(stage_df_b)
                x_diff = wx_b[0] - wx_a[0]
                y_diff = wy_b[0] - wy_a[0]
                z_diff = wz_b[0] - wz_a[0]
                # Define the starting points of the vectors
                x_start = np.array(wx_a)  # X-coordinates of starting points
                y_start = np.array(wy_a)  # Y-coordinates of starting points
                z_start = np.array(wz_a)  # Z-coordinates of starting points
                # Calculate the end points of the vectors
                x_end = np.array(x_start) + np.array(x_diff)
                y_end = np.array(y_start) + np.array(y_diff)
                z_end = np.array(z_start) + np.array(z_diff)

                all_coords = np.vstack((all_coords, np.array([x_start[0], y_start[0], z_start[0]])))
                all_coords = np.vstack((all_coords, np.array([x_end[0], y_end[0], z_end[0]])))

                marker_col = color_val
                color_val = 'lightblue'
                marker_specs = dict(size=10, color=color_val, opacity=0.7)
                if di == 1:
                    marker_specs = dict(size=[20,10], color=['Green', color_val], opacity=1, line=dict(color=['Green', color_val], width=15))
                elif di == len(days_ls)-1:
                   marker_specs = dict(size=[10,20], color=[color_val, 'Red'], opacity=1, line=dict(color=[color_val, 'Red'], width=15))

                # Add the vectors as lines from the starting points to the end points
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_start[0], x_end[0]],
                        y=[y_start[0], y_end[0]],
                        z=[z_start[0], z_end[0]],
                        mode='lines+markers',
                        line=dict(color='Black', width=6),
                        marker=marker_specs,
                        name=f'Vector {di}'
                    ),
                    row=1, col=ss_idx+1
                )

        fig.update_layout(autosize=True)
        fig.update_layout(title_text=f"Daily Displacement of Amplitude-Weighted Spike Centroid<br>{pat_id}", showlegend=False)
        # Define the camera settings
        center_dict = {'x': 0, 'y': 0, 'z': 0}
        eye_dict = {'x': 2, 'y': 2, 'z': 2}
        projection_dict = {'type': 'perspective'} # perspective, orthographic
        up_dict = {'x': 0, 'y': 0, 'z': 1}
        camera = dict(center=center_dict, eye=eye_dict, projection=projection_dict, up=up_dict)
        for i in range(len(stages_names_ls)):
            fig.update_layout(**{f'scene{i+1}': dict(camera=camera)})

        alb = 1
        axis_limits = dict(
            xaxis=dict(nticks=10, range=[np.min(all_coords[:,0])-alb, np.max(all_coords[:,0])+alb], title='X Axis'),
            yaxis=dict(nticks=10, range=[np.min(all_coords[:,1])-alb, np.max(all_coords[:,1])+alb], title='Y Axis'),
            zaxis=dict(nticks=10, range=[np.min(all_coords[:,2])-alb, np.max(all_coords[:,2])+alb], title='Z Axis'),
            aspectratio=dict(x=1, y=1, z=1)
        )

        for i in range(len(stages_names_ls)):
            fig.update_layout(**{f'scene{i+1}': axis_limits})
            
        fig.show()
        out_images_path = self.output_path /"Images/Trajectory_plus_SOZ/"
        os.makedirs(out_images_path, exist_ok=True)
        fig_fpath = out_images_path / f"{pat_id}_AllDays_SpkAmp_wAvgCntct_Trajectory_plus_SOZ.html"    
        fig.write_html(fig_fpath)

        fig.update_layout(autosize=True,width=2048,height=1024)
        fig_fpath = out_images_path / f"{pat_id}_AllDays_SpkAmp_wAvgCntct_Trajectory_plus_SOZ.jpg"
        fig.write_image(fig_fpath)
        pass