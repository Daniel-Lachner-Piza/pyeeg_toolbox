import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict
from pyeeg_toolbox.persyst.spike_cumulator import SpikeCumulator
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO
from scipy.signal import find_peaks, peak_prominences
from studies_info import fr_ILAES2025_patients

class SpikeAmplitudeAnalyzer:
    """
    A class for analyzing the spike actvity across sleep stages based on the average spike amplitude per channel.
    """

    def __init__(self, 
                 pat_id:str=None,
                 ieeg_data_path:str=None, 
                 sleep_data_path:str=None, 
                 ispikes_data_path:str=None, 
                 sleep_stages_map:Dict[int, str]={0: "Unknown", 1: "N3", 2: "N2", 3:"N1", 4:"REM", 5:"Wake"},
                 )->None:
        """
        Initialize the SpikeAmplitudeAnalyzer class..

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
        self.sleep_stages_map = sleep_stages_map

        if self.sleep_data_path is None:
            self.sleep_data_path = self.ieeg_data_path
        if self.ispikes_data_path is None:
            self.ispikes_data_path = self.ieeg_data_path

        self.eeg_file_extension = ".lay"
        self.pat_files_ls = None
        self.spike_cumulator = None


    def run(self, file_extension:str='.lay', mtg_t:str='ir', plot_ok:bool=False)->None:
        """
        This function orchestrates the entire spike analysis process.

        It performs the following steps:
        1. Retrieves all files with the specified extension (default is '.lay'), from the specified directory.
        2. Calculates the total duration of each sleep stage for the patient.
        3. Cumulates spike events for each sleep stage, using a matrix where each row corresponds to an EEG channel.

        Parameters:
        file_extension (str): The file extension to filter for. Default is '.lay'.
        mtg_t (str): The montage type to use for EEG data reading. Default is 'ir'. Options:
            'sr' = Scalp Referential
            'sb' = Scalp Bipolar
            'ir' = Intracranial Referential
            'ib' = Intracranial Bipolar
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """
        self.get_files_in_folder(file_extension)
        self.get_sleep_stages_duration_sec()
        self.get_channel_avg_spike(mtg_t, plot_ok) # 'sr', 'sb', 'ir', 'ib'

    
    def get_files_in_folder(self, file_extension:str='.lay') -> None:
        """
        This function retrieves all files with a specific extension from a given directory.

        Parameters:
        file_extension (str): The file extension to filter for. Default is '.lay'.

        Returns:
        None
        """
        self.eeg_file_extension = file_extension
        self.pat_files_ls = [fn for fn in self.ieeg_data_path.glob(f"*{self.eeg_file_extension}")]
        # Check if any files were found
        assert len(self.pat_files_ls)>0, f"No {self.eeg_file_extension} files in folder {self.ieeg_data_path}"



    def read_sleep_stages_data(self, eeg_filepath:str=None) -> pd.DataFrame:
        """
        Read sleep stages data from a CSV file.
        The CSV file is assumed to be in the same directory as the EEG file.
        The CSV file is assumed to be named exactly the same as the EEG file but replacing the filexnetsion for the suffix '_ScalpSleepStages.csv': 
        The format of the CSV file is as follows:
            - Ignore first 7 lines
            - On the 8th row place the column names: ClockDateTime, Time, I1_1, I2_1, I3_1
            - Place sleep stages in the column named I1_1
 
        Parameters:
        eeg_filepath (str): The path to the EEG file.

        Returns:
        pd.DataFrame: A pandas DataFrame containing the sleep stages data. The DataFrame will have columns for 'Time' and 'I1_1', where 'I1_1' represents the sleep stage code.
        """
        sleep_data_filepath = self.sleep_data_path / eeg_filepath.name.replace(self.eeg_file_extension, '_ScalpSleepStages.csv')
        sleep_data_df = pd.read_csv(sleep_data_filepath, skiprows=7)
        return sleep_data_df

    
    def read_spike_data(self, eeg_filepath:str=None)->pd.DataFrame:
        """
        Read spike detections data from a CSV file.
        The CSV file is assumed to be in the same directory as the EEG file.
        The CSV file is assumed to be named exactly the same as the EEG file but replacing the filexnetsion for the suffix '.sd4.csv':
        The format of the CSV file is as follows:
            - The column names are: Type, Time, Channel, Group, Perception, Sign, Duration, Height, Angle

        Parameters:
        eeg_filepath (str): The path to the EEG file.

        Returns:
        pd.DataFrame: A pandas DataFrame containing the spike detections data. The DataFrame will have columns for 'Time', 'Channel' and 'Sign', where 'Sign' denotes the polarity of the spike.
        """
        spike_data_filepath = self.ispikes_data_path / eeg_filepath.name.replace(self.eeg_file_extension, '.sd4.csv')
        spike_data_df = pd.read_csv(spike_data_filepath, skiprows=0)
        return spike_data_df

    def undersample_signal(self, signal:np.ndarray, fs:int)->np.ndarray:
        return signal[np.linspace(start=0, stop=len(signal), num=fs, dtype=int, endpoint=False)]

    def get_spike_features(self, spike_sig:np.ndarray, fs:int, plot_ok:bool=False)->tuple[float,float]:
        """
        Returns the amplitude and frequency of the spike signal.
        """
        spike_sig *= 1000 # transform to mV 
        reduced_wdw_dur_s = 0.2
        reduced_spike_sig = spike_sig[int(fs*reduced_wdw_dur_s):int(-reduced_wdw_dur_s*fs)]
        inv_red_spike_sig = reduced_spike_sig*-1
        mean_ampl = np.mean(spike_sig)
        all_peaks, _ = find_peaks(inv_red_spike_sig,prominence=0)
        all_proms = peak_prominences(inv_red_spike_sig, all_peaks)[0]
        if len(all_proms) == 0:
            return np.nan, np.nan

        min_prom_th = np.max(all_proms)*0.3
        relev_peaks, _  = find_peaks(inv_red_spike_sig, prominence=min_prom_th)
        relev_peaks = np.sort(relev_peaks)

        reduced_spike_mid = int(len(reduced_spike_sig)/2)
        lpks = relev_peaks[(reduced_spike_mid - np.array(relev_peaks))>0]
        rpks = relev_peaks[(reduced_spike_mid - np.array(relev_peaks))<0]

        # Keep looking for spike in the left side of the spike peak
        if len(lpks) == 0:
            temp_prom_th = min_prom_th
            for _ in range(5):
                temp_prom_th = temp_prom_th - (temp_prom_th*0.5)
                temp_peaks, _  = find_peaks(inv_red_spike_sig, prominence=temp_prom_th)
                temp_peaks = np.sort(temp_peaks)
                lpks = temp_peaks[(reduced_spike_mid - np.array(temp_peaks))>0]
                if len(lpks) > 0:
                    break

        # Keep looking for spike in the right side of the spike peak
        if len(rpks) == 0:
            temp_prom_th = min_prom_th
            for _ in range(5):
                temp_prom_th = temp_prom_th - (temp_prom_th*0.5)
                temp_peaks, _  = find_peaks(inv_red_spike_sig, prominence=temp_prom_th)
                temp_peaks = np.sort(temp_peaks)
                rpks = temp_peaks[(reduced_spike_mid - np.array(temp_peaks))<0]
                if len(rpks) > 0:
                    break
        
        red_spike_start = 0
        red_spike_end = len(reduced_spike_sig)
        if len(lpks) > 0: 
            red_spike_start = lpks[len(lpks)-1]
        
        if len(rpks) > 0:
            #red_spike_end = rpks[len(rpks)-1]
            red_spike_end = rpks[0]
                
        ampl = np.max(reduced_spike_sig[red_spike_start:red_spike_end+1])-np.min(reduced_spike_sig[red_spike_start:red_spike_end+1])
        dur_s = (red_spike_end-red_spike_start+1)/fs
        freq = 1/dur_s

        if plot_ok:
            plt.figure(figsize=(10,6))
            
            plt.subplot(1, 4, 1)
            time_vec = np.arange(len(spike_sig))/fs
            plt.plot(time_vec, spike_sig, '-k', linewidth=1)
            plt.plot([np.mean(time_vec)]*2, [np.min(spike_sig), np.max(spike_sig)], '--r', linewidth=1)
            plt.xlim(np.min(time_vec), np.max(time_vec))
            plt.ylabel("Amplitude(mV)")
            plt.xlabel("Time(s)")
            plt.title("Spike Wdw")

            plt.subplot(1, 4, 2)
            time_vec = reduced_wdw_dur_s + np.arange(len(reduced_spike_sig))/fs
            plt.plot(time_vec, reduced_spike_sig, '-k', linewidth=1)
            plt.plot([np.mean(time_vec)]*2, [np.min(spike_sig), np.max(spike_sig)], '--r', linewidth=1)
            plt.xlim(np.min(time_vec), np.max(time_vec))
            plt.title("Reduced Spike Wdw")

            plt.subplot(1, 4, 3)
            time_vec = reduced_wdw_dur_s + np.arange(len(inv_red_spike_sig))/fs
            plt.plot(time_vec, inv_red_spike_sig, '-k', linewidth=1)
            plt.plot([np.mean(time_vec)]*2, [np.min(spike_sig), np.max(spike_sig)], '--r', linewidth=1)
            plt.plot(time_vec[relev_peaks], inv_red_spike_sig[relev_peaks], "x")
            plt.xlim(np.min(time_vec), np.max(time_vec))
            plt.plot(np.zeros_like(inv_red_spike_sig)+mean_ampl, "--", color="gray")
            plt.title("Inverted Reduced Spike Wdw\nDetected Peaks")

            ax = plt.subplot(1, 4, 4)
            time_vec = reduced_wdw_dur_s + np.arange(len(reduced_spike_sig))/fs
            plt.plot(time_vec, reduced_spike_sig, '-k', linewidth=1)
            plt.plot(time_vec[relev_peaks], reduced_spike_sig[relev_peaks], "x")
            plt.plot(time_vec[[reduced_spike_mid]*2], [reduced_spike_sig[reduced_spike_mid], reduced_spike_sig[reduced_spike_mid]-ampl], '-m')
            plt.plot([time_vec[red_spike_start],time_vec[red_spike_end]] , [reduced_spike_sig[red_spike_start],reduced_spike_sig[red_spike_end]], 'xr')
            plt.plot([time_vec[red_spike_start],time_vec[red_spike_end]] , [reduced_spike_sig[reduced_spike_mid]]*2, '-m')
            plt.plot([time_vec[red_spike_start],time_vec[red_spike_end]] , [reduced_spike_sig[reduced_spike_mid]-ampl]*2, '-m')
            plt.xlim(np.min(time_vec), np.max(time_vec))
            plt.title("Reduced Spike Wdw\nDetected Peaks")

            text_str = f'Amp-PtP:{ampl:.3f}mV\n Freq:{freq:.1f}Hz'
            ax.text(0.70, 0.95, text_str, transform=ax.transAxes, fontsize=12, va='top', ha='left')

            # plt.suptitle(f"PatientID {this_pat_eeg_file_path.name}\n SpikeNr:{spike_idx+1}/{len(spike_data_df)}\nSpikeCh:{spike_eeg_ch_name}, SleepStage:{spike_sleep_stage_name}, Polarity: {spike_polarity}")
            # Display plot and wait for user input.
            #plt.get_current_fig_manager().full_screen_toggle()
            plt.waitforbuttonpress()
            plt.close()
            pass

        return ampl, freq


    def initialize_spike_cumulator(self, mtg_t:str='ir')->None:
        """
        Initialize the SpikeCumulator object to accumulate spike signals for each EEG channel and sleep stage.

        Parameters:
        mtg_t (str): The montage type to use for EEG data reading. Default is intracranial referential='ir'.

        Returns:
        None
        """
        this_pat_eeg_file_path = ""
        try:
            assert len(self.pat_files_ls) > 0, f"No files found in folder {self.ieeg_data_path}"

            all_files_ch_names_ls = []
            for record_idx,this_pat_eeg_file_path in enumerate(self.pat_files_ls):
                eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t=mtg_t)
                all_files_ch_names_ls.extend(eeg_reader.ch_names)
                pass
                # Initialize structure to count number of sample windows and accumulate these sample windows
                #print(f"Progress:{(record_idx+1)/len(self.pat_files_ls)*100}%")

        except Exception as e:
            print(e)
            if self.pat_files_ls==None:
                print(f"iEEG files list is indetermined, try calling get_files_in_folder() before calling characterize_channel_spike_amplitude()")
            elif len(self.pat_files_ls)==0:
                print(f"No files found in folder {self.ieeg_data_path}")
            else:
                print(f"Error reading iEEG file {this_pat_eeg_file_path}")

        sleep_stages_ls = list(self.sleep_stages_map.values())
        all_files_unique_ch_names_ls = list(set(all_files_ch_names_ls))
        self.spike_cumulator = SpikeCumulator(eeg_channels_ls=all_files_unique_ch_names_ls, sleep_stage_ls=sleep_stages_ls, sig_wdw_dur_s=1, sig_wdw_fs=128)

        return None


    def get_channel_avg_spike(self, mtg_t:str='ir', plot_ok:bool=False)->SpikeCumulator:
        """
        This function cumulates spike events for each EEG channel and sleep stage.

        Parameters:
        mtg_t (str): The montage type to use for EEG data reading. Default is intracranial referential='ir'.
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """
        assert len(self.pat_files_ls) > 0, f"No files found in folder {self.ieeg_data_path}"

        # Initialize structure to count number of sample windows and accumulate these sample windows
        self.initialize_spike_cumulator(mtg_t=mtg_t)

        this_pat_eeg_file_path = ""           
        for record_idx in np.arange(start=0, stop=len(self.pat_files_ls)):
            this_pat_eeg_file_path = self.pat_files_ls[record_idx]
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t=mtg_t)
            #fs = eeg_reader.fs

            # Read sleep data and spike detections
            sleep_data_df = self.read_sleep_stages_data(this_pat_eeg_file_path)
            spike_data_df = self.read_spike_data(this_pat_eeg_file_path)
            spike_data_df = spike_data_df.sort_values(by=['Time'], ascending=True)
            # Clean spike channel using same method used for eeg channel cleaning
            spike_data_df.Channel = eeg_reader.clean_channel_labels(spike_data_df.Channel.values.tolist())

            for spike_idx in range(len(spike_data_df)):

                spike_center_sec = spike_data_df.at[spike_idx,'Time']
                spike_wdw_start_sec = (spike_center_sec-0.5)
                spike_wdw_start_sample = int((spike_center_sec-0.5)*eeg_reader.fs)
                spike_wdw_end_sample = int((spike_center_sec+0.5)*eeg_reader.fs)
                if spike_wdw_start_sample < 0 or spike_wdw_end_sample > eeg_reader.n_samples-1:
                    continue

                spike_sleep_stage_code = sleep_data_df.I1_1[sleep_data_df.Time==int(spike_center_sec)].to_numpy().flatten()
                assert len(spike_sleep_stage_code)==1, "Could not assign a sleep stage to spike"
                if np.isnan(spike_sleep_stage_code):
                    continue
                spike_sleep_stage_name = self.sleep_stages_map[int(spike_sleep_stage_code[0])]
                spike_polarity = spike_data_df.at[spike_idx,'Sign']>0

                spike_det_ch_name = spike_data_df.at[spike_idx,'Channel']
                spike_eeg_ch_name = [ch for ch in eeg_reader.ch_names if spike_det_ch_name.lower()==ch.lower()]
                assert len(spike_eeg_ch_name)==1, "Could not assign a channel to spike"
                spike_eeg_ch_name = spike_eeg_ch_name[0]
                spike_eeg_ch_idx = eeg_reader.ch_names.index(spike_eeg_ch_name)

                # Get the EEG segment containing the spike
                spike_wdw = eeg_reader.get_data(picks=spike_eeg_ch_idx, start=spike_wdw_start_sample, stop=spike_wdw_end_sample).flatten()
                if not spike_polarity:
                    spike_wdw = spike_wdw*-1

                # Undersample the EEG segment containing the spike
                fs_us = self.spike_cumulator.get_undersampling_frequency()
                spike_wdw_us = self.undersample_signal(spike_wdw, fs_us)

                #spike_feats=self.get_spike_features(spike_wdw_us, fs_us, True)
                spike_feats = (0,0)
                spike_ampl = spike_feats[0]
                spike_freq= spike_feats[1]
                if np.isnan(spike_feats[0]):
                    continue

                self.spike_cumulator.add_spike(spike_sleep_stage_name, spike_eeg_ch_name, spike_wdw_us, spike_ampl, spike_freq)

                if plot_ok or np.isnan(spike_feats[0]):
                    plt.figure(figsize=(10,6))
                    plt.subplot(1, 3, 1)
                    time_vec = spike_wdw_start_sec+np.arange(len(spike_wdw))/eeg_reader.fs
                    plt.plot(time_vec, spike_wdw, '-k', linewidth=1)
                    plt.plot([np.mean(time_vec)]*2, [np.min(spike_wdw), np.max(spike_wdw)], '--r', linewidth=1)
                    plt.xlim(np.min(time_vec), np.max(time_vec))

                    plt.subplot(1, 3, 2)
                    time_vec = spike_wdw_start_sec+np.arange(len(spike_wdw_us))/fs_us
                    plt.plot(time_vec, spike_wdw_us, '-k', linewidth=1)
                    plt.plot([np.mean(time_vec)]*2, [np.min(spike_wdw_us), np.max(spike_wdw_us)], '--r', linewidth=1)
                    plt.xlim(np.min(time_vec), np.max(time_vec))

                    plt.subplot(1, 3, 3)
                    abs_spike_wdw_us = np.abs(spike_wdw_us)
                    time_vec = spike_wdw_start_sec+np.arange(len(spike_wdw_us))/fs_us
                    plt.plot(time_vec, abs_spike_wdw_us, '-k', linewidth=1)
                    plt.plot([np.mean(time_vec)]*2, [np.min(abs_spike_wdw_us), np.max(abs_spike_wdw_us)], '--r', linewidth=1)
                    plt.xlim(np.min(time_vec), np.max(time_vec)) 

                    plt.suptitle(f"PatientID {this_pat_eeg_file_path.name}\n SpikeNr:{spike_idx+1}/{len(spike_data_df)}\nSpikeCh:{spike_eeg_ch_name}, SleepStage:{spike_sleep_stage_name}, Polarity: {spike_polarity}")

                    # Display plot and wait for user input.
                    #plot_ok = not plt.waitforbuttonpress()
                    plt.waitforbuttonpress()
                    plt.close()
                    #if not plot_ok:
                    #    return None

            print(f"Progress:{(record_idx+1)/len(self.pat_files_ls)*100}%")

        return self.spike_cumulator


    def get_sleep_stages_duration_sec(self)->dict:
        """
        Calculate and print the total duration of each sleep stage for the patient.

        Parameters:
        None

        Returns:
        dict: A dictionary where keys are sleep stage names and values are the total duration of each sleep stage in seconds.
        """
        sleep_stage_secs_counter_dict = {v:0 for v in self.sleep_stages_map.values()}
        sleep_stage_secs_counter_dict['NaN'] = 0
        for i in range(len(self.pat_files_ls)):
            this_pat_eeg_file_path = self.pat_files_ls[i]
            pat_sleep_data_path = self.sleep_data_path / this_pat_eeg_file_path.name.replace(self.eeg_file_extension, '_ScalpSleepStages.csv')
            sleep_data_df = pd.read_csv(pat_sleep_data_path, skiprows=7)
            sleep_stage_secs_counter_dict['NaN'] += sleep_data_df.I1_1.isna().sum()
            for sleep_key in self.sleep_stages_map.keys():
                sleep_stage_secs_counter_dict[self.sleep_stages_map[sleep_key]] += np.sum(sleep_data_df.I1_1==sleep_key)

        days_analyzed = 0
        print("\nTotal Duration of analyzed Sleep Stages:")
        for k,v in sleep_stage_secs_counter_dict.items():
            print(f"{k}={v/3600:.2f} hours")
            days_analyzed += v
        days_analyzed /= (3600*24)
        print(f"Days analyzed = {days_analyzed:.2f}\n")

        return sleep_stage_secs_counter_dict


    def save_spike_cumulator(self, filepath:str=None):
        """
        Save the SpikeCumulator object to a file using pickle serialization.

        Parameters:
        filepath (str): The path to the file where the SpikeCumulator object will be saved.

        Returns:
        None
        """
        if self.spike_cumulator is not None:
            os.makedirs(filepath.parent, exist_ok=True)
            with open(filepath, 'wb') as handle:
                pickle.dump(self.spike_cumulator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_spike_cumulator(self, filepath:str=None) -> None:
        """
        Load a SpikeCumulator object from a file using pickle deserialization.

        Parameters:
        filepath (str): The path to the file where the SpikeCumulator object is saved.

        Returns:
        None
        """
        with open(filepath, 'rb') as handle:
            self.spike_cumulator = pickle.load(handle)
        return self.spike_cumulator
    
    def MinMaxScaler(self, data):
        return (data - np.min(data))/(np.max(data)-np.min(data))


if __name__ == '__main__':


    # Define directory to save the cumulated spike signals
    output_path = Path(os.getcwd()) / "Output"
    os.makedirs(output_path, exist_ok=True)

    study_info = fr_ILAES2025_patients()

    for pat_id in study_info.patients.keys():

        print(pat_id)

        pat_data_path = study_info.eeg_data_path / pat_id
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)

        spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=False)

        spike_cumulator_fn = output_path / f"{pat_id}_SpikeCumulator.pickle"
        spike_amplitude_analyzer.save_spike_cumulator(filepath=spike_cumulator_fn)
