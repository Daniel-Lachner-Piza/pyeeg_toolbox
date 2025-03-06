import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from joblib import Parallel, delayed
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from pyeeg_toolbox.persyst.an_avg_wdw_amplitude import AverageWdwAnalyzer
from pyeeg_toolbox.persyst.an_spike_wdw_avg_by_day import DailySpikeWdwAnalyzer
from pyeeg_toolbox.persyst.an_avg_wdw_amplitude_vectorized import VectorizedAvgWdwAnalyzer

from collections import defaultdict

from studies_info import fr_ILAES2025_patients
import pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude as spk_plt
import pyeeg_toolbox.persyst.an_plot_avg_wdw_amplitude as wdw_plt

FORCE_RECALC = False

# Define directory to save the cumulated spike signals
output_path = Path(os.getcwd()) / "Output"
os.makedirs(output_path, exist_ok=True)

study_info = fr_ILAES2025_patients()

def analyze_spike_wdws_vectorized(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    pat_coords_path = study_info.channel_coordinates_data_path
    pat_szr_info_path = study_info.seizure_info_data_path
    spike_amplitude_analyzer = VectorizedAvgWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path, ch_coordinates_data_path=pat_coords_path, szr_info_data_path=pat_szr_info_path, output_path=output_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', force_recalc=FORCE_RECALC)
    pass


output_path = Path(os.getcwd()) / "Vectorized_WdwAn_Output"
os.makedirs(output_path, exist_ok=True)
# Analyze data and plot results
results = Parallel(n_jobs=1)(delayed(analyze_spike_wdws_vectorized)(study_info, pat_id) for pat_id in study_info.patients.keys())
