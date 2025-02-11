import numpy as np

class AvgWdwCumulator():
    """
    A class to accumulate the spike signals for each channel and having each spike centered on the max amplitude
    """

    def __init__(self, 
                 eeg_channels_ls:list=None,
                 sleep_stage_ls:list=None,
                 sig_wdw_dur_s:float=1.0,
                 sig_wdw_fs:float=64.0,
                 ):
        """
        Initializes the class.

        Args:
            _
            _
        """
        self.eeg_channels_ls = [chname.lower() for chname in eeg_channels_ls]
        self.sleep_stage_ls = sleep_stage_ls
        self.sig_wdw_dur_s = sig_wdw_dur_s
        self.sig_wdw_fs = sig_wdw_fs
        nr_chs = len(self.eeg_channels_ls)
        self.spike_cum_dict = {stage:np.zeros(shape=(nr_chs, int(sig_wdw_dur_s*sig_wdw_fs))) for stage in self.sleep_stage_ls}   
        self.spike_ampl = {stage:np.zeros(shape=(nr_chs, 1)) for stage in self.sleep_stage_ls}  
        self.spike_freq = {stage:np.zeros(shape=(nr_chs, 1)) for stage in self.sleep_stage_ls}
        self.nr_spikes = {stage:np.zeros(shape=(nr_chs, 1)) for stage in self.sleep_stage_ls}

        for k in self.spike_cum_dict.keys():
            assert self.spike_cum_dict[k].shape[0] == nr_chs, f"Spike cumulator has wrong nr. channels for stage {k}"
            assert self.spike_cum_dict[k].shape[1] == int(self.sig_wdw_fs*self.sig_wdw_dur_s), f"Spike cumulator has wrong nr. samples for stage {k}"

        pass

    def get_undersampling_frequency(self):
        return self.sig_wdw_fs

    def get_channels_ls(self):
        return self.eeg_channels_ls
    
    def get_ch_idx(self, ch_name):
        chidx = self.eeg_channels_ls.index(ch_name.lower())
        return chidx
    
    def add_spike(self, sleep_stage:str=None, ch_name:str=None, avg_spike_signal:np.array=None, amplitude:float=0, frequency:float=0, nr_spikes:int=0):
        """
        Adds a spike signal to the spike counter and cumulator.
        """
        try:
            ch_idx = self.get_ch_idx(ch_name)
            self.spike_cum_dict[sleep_stage][ch_idx] = avg_spike_signal
            self.spike_ampl[sleep_stage][ch_idx]= amplitude
            self.spike_freq[sleep_stage][ch_idx]= frequency
            self.nr_spikes[sleep_stage][ch_idx]= nr_spikes

        except Exception as e:
            print(e)
            print(f"Trying to add a spike to channel {ch_name}, whcih doesn't exist in the spike cumulator, check channel name being added or initialization of spike cumulator")
        
        pass