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

from studies_info import fr_ILAES2025_patients, ACH_Pediatric_Patients, ACH_Pediatric_Patients_Spike_Drive
import pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude as spk_plt
import pyeeg_toolbox.persyst.an_plot_avg_wdw_amplitude as wdw_plt

FORCE_RECALC = False

def analyze_spike_wdws_vectorized(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    pat_coords_path = study_info.channel_coordinates_data_path
    pat_szr_info_path = study_info.seizure_info_data_path
    spike_amplitude_analyzer = VectorizedAvgWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path, ch_coordinates_data_path=pat_coords_path, szr_info_data_path=pat_szr_info_path, output_path=output_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', force_recalc=FORCE_RECALC)

def analyze_sleep_stages(study_info):
    all_pats_sleep_stages_df = pd.DataFrame()
    for pat_id in study_info.patients.keys():
        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_amplitude_analyzer = VectorizedAvgWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        pat_sleep_df = spike_amplitude_analyzer.analyze_sleep(file_extension='.lay')

        all_pats_sleep_stages_df = pd.concat([all_pats_sleep_stages_df, pat_sleep_df], ignore_index=True)
        pass

    # sns.set(style="whitegrid")
    # ax= sns.barplot(all_pats_sleep_stages_df, x='PatID', y='StageDurationH', hue='Stage')
    # for container in ax.containers:
    #     ax.bar_label(container, fontsize=10, rotation=90, label_type='edge', padding=3, color='black', fmt='%.2f')
    # plt.title('Sleep Stages Duration by Patient')
    # plt.show()
    # #plt.waitforbuttonpress()
    # all_pats_sleep_stages_df.to_csv(output_path / "ALL_Patients_Sleep_Stages.csv", index=False)

    return all_pats_sleep_stages_df



if __name__ == "__main__":

    # Define directory to save the cumulated spike signals
    output_path = Path(os.getcwd()) / "Output"
    os.makedirs(output_path, exist_ok=True)

    study_info = fr_ILAES2025_patients()
    # study_info = ACH_Pediatric_Patients()
    # study_info = ACH_Pediatric_Patients_Spike_Drive()

    output_path = Path(os.getcwd()) / "Vectorized_WdwAn_Output"
    os.makedirs(output_path, exist_ok=True)


    #sleep_a_df = analyze_sleep_stages(study_info)
    results = Parallel(n_jobs=1)(delayed(analyze_spike_wdws_vectorized)(study_info, pat_id) for pat_id in study_info.patients.keys())


    pass