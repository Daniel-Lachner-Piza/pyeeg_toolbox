import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
import scikit_posthocs as sp
#import statsmodels as sm
import sys

from PIL import Image
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay,roc_curve, roc_auc_score
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from studies_info import fr_ILAES2025_patients, ACH_Pediatric_Patients_All
from statsmodels.stats.anova import AnovaRM 
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch


from imblearn.over_sampling import RandomOverSampler, SMOTE

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")

FIGSIZE = (16, 8)


# Utility function to convert hex color to rgb string
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r,g,b)

STAGES_COLORS_SELECT = {
    "Sleep": "#FF9532FF",    # Orange
    "N1": "#FAE163",    # Light gold
    "N2": "#29E8B2",    # Turquoise
    "N3": "#4CA9EE",    # Sky blue
    "REM": "#2F4571",   # Dark blue
    "Wake": "#B93D4E",  # Light red
    "Unknown": "#808080" # Gray
}

STAGES_COLORS = {stage_name:hex_to_rgb(color) for stage_name,color in STAGES_COLORS_SELECT.items()}

class Spike_Activity_Analyzer:
    """
    Class to analyze spike activity in EEG data, including reading data, handling outliers, scaling, and plotting results.
    """

    def __init__(self, study_name:str='Spike_Activity_Study', characterization_datapath:str=None, stages_spikes_duration_rate_datapath:str=None, pats_ls:list[str]=None, sleep_stages_ls:list[str]=None, stages_colors:dict=None, images_output_path:str=None, szr_cnt_df:pd.DataFrame=None):
        self.study_name = study_name
        self.characterization_datapath = characterization_datapath
        self.stages_spikes_duration_rate_datapath = stages_spikes_duration_rate_datapath
        self.pats_ls = pats_ls
        self.sleep_stages_ls = sleep_stages_ls
        self.stages_colors = stages_colors
        self.images_output_path = images_output_path
        self.szr_cnt_df = szr_cnt_df

    def bootstrap_confidence_interval_univariate(self, data, func=np.mean, confidence_level=0.95, n_resamples=10000):
        """
        Calculate the bootstrap confidence interval for a given data set and function.
        
        Parameters:
        - data: The data to bootstrap.
        - func: The function to apply to the data (default is np.mean).
        - confidence_level: The confidence level for the interval (default is 0.95).
        - n_resamples: The number of bootstrap resamples (default is 10000).
        
        Returns:
        - A tuple containing the lower and upper bounds of the confidence interval.
        """
        boot_means = func(np.random.choice(data, (n_resamples, len(data)), replace=True), axis=1)

        # 95% confidence interval (2.5th and 97.5th percentiles)
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        ci_range = (ci_lower, ci_upper)

        # Use scipy's bootstrap for a more robust implementation
        # res = bootstrap((data,), func, confidence_level=confidence_level, n_resamples=n_resamples, method='percentile')
        # ci_range = (res.confidence_interval.low, res.confidence_interval.high)
        return ci_range

    def read_patient_spike_data(self, stage_duration_spike_rate_df: pd.DataFrame=None):
        nr_pats = len(self.pats_ls)
        # Concatenate data from all patients
        spike_data_df = pd.DataFrame()
        for pdata_fn in self.pats_ls:
            data_fpath = self.characterization_datapath / f"{pdata_fn}_AvgSpikeWdwActivity.csv"
            print(data_fpath)
            try:
                pdata_df = pd.read_csv(data_fpath)
                pdata_df['Patient'] = pdata_fn.replace('_AvgSpikeWdwActivity.csv', '')

                pdata_df.loc[pdata_df.Stage.isna().to_numpy(), 'Stage'] = 'NaN_SleepStage'
                pdata_df.loc[pdata_df.Stage.isnull().to_numpy(), 'Stage'] = 'NaN_SleepStage'
                assert pdata_df.isna().sum().sum() == 0, f"NaN values in {pdata_fn}"
                assert pdata_df.isnull().sum().sum() == 0, f"Null values in {pdata_fn}"

                # Add weighted average spike activity for Sleep, i.e. the sum of all sleep stages N3, N2, N1, REM weighted by their duration
                pat_stage_dur = stage_duration_spike_rate_df[stage_duration_spike_rate_df.PatID.str.fullmatch(pdata_fn, case=False)].reset_index(drop=True).copy()
                pat_stage_dur = pat_stage_dur[pat_stage_dur.Stage.str.fullmatch('N3|N2|N1|REM', case=False)]
                pdata_sleep_df = pdata_df[pdata_df.Stage.str.fullmatch('N3|N2|N1|REM', case=False)]
                sleep_chann_data = {col_name:[] for col_name in pdata_df.columns.tolist()}
                for chname in pdata_sleep_df.Channel.unique():
                    chann_sel = pdata_sleep_df.Channel.str.fullmatch(chname, case=False)
                    assert np.sum(chann_sel) == 4, f"More than one entry per sleep stage in {pdata_fn}"
                    ch_amplitudes = pdata_sleep_df.loc[chann_sel, ['Stage','Amplitude']]
                    ch_amplitudes.sort_values(by='Stage', axis=0, ascending=True, inplace=True)
                    ch_stages_duration = pat_stage_dur[['Stage', 'StageDurM']].sort_values(by='Stage', axis=0, ascending=True)
                    assert np.sum(ch_amplitudes.Stage.values == ch_stages_duration.Stage.values) == 4, "Sleep stages do not match in Amplitude and StageDurM"
                    ch_weighted_amplitude = np.average(ch_amplitudes.Amplitude, weights=ch_stages_duration.StageDurM)
                    nr_clips_with_spikes = pdata_sleep_df.loc[chann_sel, ['NrClipsWithSpikes']].max().values[0]
                    ch_soz = pdata_sleep_df.loc[chann_sel, 'SOZ'].unique()
                    assert len(ch_soz) == 1, f"More than one SOZ value for channel {chname} in {pdata_fn}"
                    
                    sleep_chann_data['Stage'].append('Sleep')
                    sleep_chann_data['Channel'].append(chname)
                    sleep_chann_data['Amplitude'].append(ch_weighted_amplitude)
                    sleep_chann_data['NrClipsWithSpikes'].append(nr_clips_with_spikes)
                    sleep_chann_data['SOZ'].append(ch_soz[0])
                    sleep_chann_data['Patient'].append(pdata_fn.replace('_AvgSpikeWdwActivity.csv', ''))

                    pass
                pdata_df = pd.concat([pdata_df, pd.DataFrame(sleep_chann_data)], ignore_index=True)

                spike_data_df = pd.concat([spike_data_df, pdata_df])
            except Exception as e:
                print(f"Failed to load {data_fpath}: {e}")

        spike_data_df['SOZ'] = spike_data_df['SOZ'].astype('string')
        spike_data_df.loc[spike_data_df.SOZ=='1', 'SOZ'] = 'SOZ'
        spike_data_df.loc[spike_data_df.SOZ=='0', 'SOZ'] = 'Non-SOZ'

        entries_per_stage = spike_data_df.Stage.value_counts()
        print(f"Entries per stage: {entries_per_stage}")
        assert entries_per_stage.unique().shape[0] == 1, "Not all patients have the same number of entries per stage"

        spike_data_df = spike_data_df[spike_data_df.Stage != 'NaN_SleepStage']

        spike_data_df.reset_index(drop=True, inplace=True)

        return spike_data_df

    def read_stages_duration_and_spike_rates(self):
        # Concatenate data from all patients
        nr_pats = len(self.pats_ls)        
        spike_occ_rate_pats_ls = [pn + "_StageSpikeOccurrenceRate.csv" for pn in self.pats_ls]
        stage_duration_spike_rate_df = pd.DataFrame()
        for pdata_fn in spike_occ_rate_pats_ls:
            data_fpath = stages_spikes_duration_rate_datapath / pdata_fn
            #print(data_fpath)
            try:
                pdata_df = pd.read_csv(data_fpath)
                pat_sleepdata_df = pdata_df[pdata_df.Stage.str.fullmatch('N3|N2|N1|REM', case=False)]
                sleep_occ_rate = np.average(pat_sleepdata_df.SpikeOccRate, weights=pat_sleepdata_df.StageDurM)
                pdata_sleep_df = {'PatID': pdata_df.PatID.unique()[0], 'Stage': 'Sleep', 'StageDurM': pat_sleepdata_df.StageDurM.sum(), 'SpikeOccRate': sleep_occ_rate}
                pdata_sleep_df = pd.DataFrame(pdata_sleep_df, index=[0])
                pdata_df = pd.concat([pdata_sleep_df, pdata_df], ignore_index=True)
                stage_duration_spike_rate_df = pd.concat([stage_duration_spike_rate_df, pdata_df])
            except:
                print(f"File {pdata_fn} not found")
        stage_duration_spike_rate_df.reset_index(drop=True, inplace=True)
        return stage_duration_spike_rate_df

    def handle_patient_outliers(self, spike_data_df:pd.DataFrame=None):

        # Remove outliers from the data
        raise_min = np.abs(spike_data_df.Amplitude.min())+0.1 # add 1 to avoid log(0)
        #spike_data_df.loc[:, 'Amplitude'] = np.log(spike_data_df.Amplitude.values+raise_min) # log transform to reduce skewness, add 1 to avoid log(0)

        pats_ls = spike_data_df.Patient.unique()
        clean_spike_data_df = pd.DataFrame()
        for pdata_fn in pats_ls:
            pdata_df = spike_data_df[spike_data_df.Patient.str.fullmatch(pdata_fn, case=False)].reset_index(drop=True).copy()

            assert pdata_df.isna().sum().sum() == 0, f"NaN values in {pdata_fn}"
            assert pdata_df.isnull().sum().sum() == 0, f"Null values in {pdata_fn}"

            # Remove outliers from the data using IQR method
            prctl_25 = np.percentile(pdata_df.Amplitude, 25.0)
            prctl_75 = np.percentile(pdata_df.Amplitude, 75.0)
            iqr = prctl_75 - prctl_25
            outliers_thresh_a = prctl_75 + 3 * iqr # 1.5 * IQR rule for outliers

            # Remove outliers from the data using z-score method
            # z_outliers_thresh = pdata_df.Amplitude.mean() + 3.0 * pdata_df.Amplitude.std()
            # amplitude_no_outliers = pdata_df.Amplitude[pdata_df.Amplitude < z_outliers_thresh]
            # outliers_thresh_b = amplitude_no_outliers.mean() + 3 * amplitude_no_outliers.std()
            outliers_thresh_b = pdata_df.Amplitude.mean() + 5 * pdata_df.Amplitude.std()
            
            # Remove outliers from the data using percentile method
            outliers_thresh_c = np.percentile(pdata_df.Amplitude, 99.0)# 2.5 * pdata_df.Amplitude.std()

            # Remove outliers using modified z-score
            mad = np.median(np.abs(pdata_df.Amplitude - np.median(pdata_df.Amplitude)))
            modified_z_scores = 0.6745 * (pdata_df.Amplitude - np.median(pdata_df.Amplitude)) / mad
            outliers_thresh_d = 3.5 * pdata_df.Amplitude.std()
            outliers_sel = modified_z_scores > outliers_thresh_d

            outliers_thresh = outliers_thresh_b

            outliers_sel = (pdata_df.Amplitude > outliers_thresh).to_numpy()
            pdata_df.loc[outliers_sel, 'Amplitude'] = outliers_thresh

            # sns.histplot(data=pdata_df, x='Amplitude', bins=100, kde=True, color='blue')
            # sns.histplot(data=pdata_df.Amplitude.values, stat='probability',alpha=0.5, label=pdata_fn)
            # plt.title(f"Patient {pdata_fn} - Spike Amplitude Distribution")
            # plt.xlabel("Amplitude (uV)")
            # plt.show()
            # #plt.waitforbuttonpress()
            # plt.close()

            clean_spike_data_df = pd.concat([clean_spike_data_df, pdata_df.copy()])

        clean_spike_data_df.reset_index(drop=True, inplace=True)

        return clean_spike_data_df

    def get_patient_scaled_spike_data(self, spike_data_df:pd.DataFrame=None):
        # Concatenate data from all patients, scale amplitude for each patient
        pats_ls = spike_data_df.Patient.unique()
        scaled_spike_data_df = pd.DataFrame()

        # Convert Amplitude to uV
        spike_data_df.loc[:, 'Amplitude'] = spike_data_df.Amplitude.values*1000*1000 # convert to uV

        for pdata_fn in pats_ls:
            pdata_df = spike_data_df[spike_data_df.Patient.str.fullmatch(pdata_fn, case=False)].reset_index(drop=True).copy()
            pdata_df.Amplitude = MinMaxScaler().fit_transform(pdata_df.Amplitude.values.reshape(-1, 1)) # MinMaxScaler, StandardScaler()
            scaled_spike_data_df = pd.concat([scaled_spike_data_df, pdata_df])

        return scaled_spike_data_df
       
    def plot_group_sleep_stage_durations_barchart(self, stage_duration_spike_rate_orig_df:pd.DataFrame=None):
        
        nr_pats = len(stage_duration_spike_rate_orig_df.PatID.unique())
        all_pats_eeg_durations = (stage_duration_spike_rate_orig_df[['PatID', 'StageDurM']].groupby('PatID').sum()/60).to_numpy().flatten()
        ci_range = self.bootstrap_confidence_interval_univariate(all_pats_eeg_durations)
        print(f"\n\nAwareness Stages Duration")
        print(f"Nr. Patients: {nr_pats}")
        print(f"All patients avg. EEG duration: {np.mean(all_pats_eeg_durations):.2f} h(IQR={np.percentile(all_pats_eeg_durations, 75) - np.percentile(all_pats_eeg_durations, 25):.2f})")
        print(f"CI Range for all patients' EEG duration: {ci_range[0]:.2f} - {ci_range[1]:.2f} hours")
        
        # Analyze Sleep Stages Duration and Confidence Intervals
        stages_ci_ranges = {}
        stages_ls = stage_duration_spike_rate_orig_df.Stage.unique().tolist()
        for stage_name in stages_ls:
            stage_sel = stage_duration_spike_rate_orig_df.Stage.str.fullmatch(stage_name, case=False)
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            all_pats_stage_durations = stage_duration_spike_rate_orig_df.loc[stage_sel, 'StageDurM'].to_numpy()/60
            ci_range = self.bootstrap_confidence_interval_univariate(all_pats_stage_durations, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = ci_range
            print(f"{stage_name}: {np.mean(all_pats_stage_durations):.2f} h (CI={ci_range[0]:.2f}-{ci_range[1]:.2f})")
            pass

        # Plot Sleep Stages Duration in two subplots becase of the large difference in duration between Wake and other stages
        fig, all_axs = plt.subplots(1, 2, figsize=FIGSIZE)
        errorbar_def = ("ci", 95)  # Percentile interval for error bars
        errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}

        # Plot Sleep Stages N3, N2, N1, REM
        stage_duration_spike_rate_df = stage_duration_spike_rate_orig_df.copy()
        to_plot_stage_names = ['Sleep', 'N3', 'N2', 'N1', 'REM', 'Wake']
        stage_duration_spike_rate_df['StageDurH'] = stage_duration_spike_rate_df.StageDurM / 60.0 # convert to hours
        to_plot_stage_sel = stage_duration_spike_rate_df.Stage.isin(to_plot_stage_names)
        stage_duration_spike_rate_df = stage_duration_spike_rate_df[to_plot_stage_sel].reset_index(drop=True).copy()
        to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]
        assert nr_pats == len(stage_duration_spike_rate_df.PatID.unique()), "More than one entry per patient"
        axs = all_axs[0]
        bp_ax = sns.barplot(data=stage_duration_spike_rate_df, x='Stage', y='StageDurH', hue='Stage',
            order=to_plot_stage_names, palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        for cont in bp_ax.containers:
            axs.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("Duration (hours)", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        axs.tick_params(axis='x', labelsize=32)
        axs.tick_params(axis='y', labelsize=32)
        axs.set_ylim(0, 41) # set y-axis limit to 60 minutes
        axs.set_title(f"{self.study_name}", fontsize=48)

        # # Plot Sleep Stages N2, N1, REM, Wake
        # axs = all_axs[1]
        # to_plot_stage_names = ['N2', 'N1', 'REM', 'Wake']
        # stage_duration_spike_rate_df = stage_duration_spike_rate_orig_df.copy()
        # stage_duration_spike_rate_df['StageDurH'] = stage_duration_spike_rate_df.StageDurM / 60.0 # convert to hours
        # to_plot_stage_sel = stage_duration_spike_rate_df.Stage.isin(to_plot_stage_names)
        # stage_duration_spike_rate_df = stage_duration_spike_rate_df[to_plot_stage_sel].reset_index(drop=True).copy()
        # to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]
        # assert nr_pats == len(stage_duration_spike_rate_df.PatID.unique()), "More than one entry per patient"
        # bp_ax = sns.barplot(data=stage_duration_spike_rate_df, x='Stage', y='StageDurH', hue='Stage', 
        #     order=to_plot_stage_names, palette=to_plot_stages_colors, ax=axs,
        #     capsize=.2,
        #     errorbar=errorbar_def,
        #     err_kws=errorbar_characteristics,
        #     linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
        #     estimator=np.mean
        #     )
        # for cont in bp_ax.containers:
        #     axs.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        # axs.set_ylabel("Duration (hours)", fontsize=32)
        # axs.set_xlabel("Sleep Stage", fontsize=32)
        # axs.tick_params(axis='x', labelsize=32)
        # axs.tick_params(axis='y', labelsize=32)
        # axs.set_ylim(0, 41) # set y-axis limit to 60 minutes

        #plt.suptitle(f"{self.study_name}", fontsize=48)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Duration_Of_Sleep_Stages.png")
        #plt.waitforbuttonpress()
        plt.close()
        
    def plot_individual_sleep_stage_durations(self, stage_duration_spike_rate_df:pd.DataFrame=None):

        patients_ls = list(stage_duration_spike_rate_df.PatID.unique())
        nr_pats = len(patients_ls)

        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM']
        to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]
        nr_plot_rows = 3
        nr_plot_cols = int(np.ceil(nr_pats/nr_plot_rows))
        fig, axs = plt.subplots(nr_plot_rows, nr_plot_cols, figsize=FIGSIZE)
        for idx, patient in enumerate(patients_ls):
            patient_data_df = stage_duration_spike_rate_df[stage_duration_spike_rate_df.PatID.str.fullmatch(patient, case=False)].reset_index(drop=True).copy()
            total_sleep_duration = np.sum(patient_data_df.StageDurM)-np.sum(patient_data_df.StageDurM[patient_data_df.Stage=='Wake'])
            sum_stages_dur_mins = []
            new_to_plot_stage_names = []
            for stage_name in to_plot_stage_names:
                stage_sel = patient_data_df.Stage.str.fullmatch(stage_name, case=False)
                assert stage_sel.sum() == 1, "More than one entry per patient"
                prctg_val = (patient_data_df.StageDurM[stage_sel].values/total_sleep_duration)*100
                sum_stages_dur_mins.append(patient_data_df.StageDurM[stage_sel].sum())
                new_stage_name = f"{stage_name}({prctg_val[0]:.0f}%)"
                new_to_plot_stage_names.append(new_stage_name)
                pass

            sum_stages_dur_perc = (np.array(sum_stages_dur_mins)/np.sum(sum_stages_dur_mins))*100
            wedgeprops = {"edgecolor" : "white", 'linewidth': 5, 'antialiased': True}
            
            axs_row = int(idx/nr_plot_cols)
            axs_col = idx%nr_plot_cols
            ax = axs[axs_row, axs_col]
            #patches, texts, pcts = ax.pie(x=sum_stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':32, 'color':"w", 'weight':'bold'}, startangle=-200)
            #patches, texts, pcts = ax.pie(x=sum_stages_dur_perc, labels=new_to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':12, 'color':"w", 'weight':'bold'}, startangle=-200)
            patches, texts = ax.pie(sum_stages_dur_perc, labels=new_to_plot_stage_names, colors=to_plot_stages_colors, startangle=-200)
            for i, patch in enumerate(patches):
                texts[i].set_color(patch.get_facecolor())
            # ax.set_ylabel("Relative Duration of Sleep Stages (%)", fontsize=32)
            ax.set_title(f"{self.pats_ls[idx]}", fontsize=12, color='black')
            pass 

        plt.suptitle(f"{self.study_name}\nSleep-Staging\nNr. Patients={nr_pats}")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Individual_Duration_Of_Sleep_Stages.png")
        #plt.waitforbuttonpress()
        plt.close()

    def plot_spike_occ_rate(self, stage_duration_spike_rate_df:pd.DataFrame=None):

        nr_pats = len(stage_duration_spike_rate_df.PatID.unique())
        print(f"\n\nSpike Occurrence Rate")
        print(f"Nr. Patients: {nr_pats}")

        # Analyze Spike Occurrence Rate
        stages_ci_ranges = {}
        for stage_name in stage_duration_spike_rate_df.Stage.unique():
            stage_sel = stage_duration_spike_rate_df.Stage.str.fullmatch(stage_name, case=False)
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            all_pats_sor = stage_duration_spike_rate_df.loc[stage_sel, 'SpikeOccRate'].to_numpy()
            ci_range = self.bootstrap_confidence_interval_univariate(all_pats_sor, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = ci_range
            # test normality
            _, p_val = stats.shapiro(all_pats_sor)
            normality = "normal"
            if p_val < 0.05:
                normality = "not normal"
            print(f"{normality} -- {stage_name} ({np.mean(all_pats_sor):.2f}, CI={ci_range[0]:.2f}-{ci_range[1]:.2f}) (Spikes / electrode / min.)")
            pass

        max_ci_limit = max([max(v) for k,v in stages_ci_ranges.items()])

        # Plot Spike Occurrence Rate
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        assert nr_pats == len(stage_duration_spike_rate_df.PatID.unique()), "More than one entry per patient"
        errorbar_def = ("ci", 95)  # Percentile interval for error bars
        errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
        bp_ax = sns.barplot(data=stage_duration_spike_rate_df, x='Stage', y='SpikeOccRate', hue='Stage',
            order=self.sleep_stages_ls, palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        for cont in bp_ax.containers:
            plt.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("Spikes / electrode / min.", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim(0, max_ci_limit*1.1)
        #axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_title(f"{self.study_name}", fontsize=48)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Average_Spike_OccRate.png")
        #plt.waitforbuttonpress()
        plt.close()
   
    def analyze_sor_stages_differences(self, stage_duration_spike_rate_df:pd.DataFrame=None):
        # Analyze Differences in Spike Occurrence Rate between Sleep Stages
        patients_ls = list(stage_duration_spike_rate_df.PatID.unique())
        nr_pats = len(patients_ls)
        print(f"Spike Occurrence Rate Differences between Sleep Stages")
        print(f"Nr. Patients: {nr_pats}")

        stages_ls = self.sleep_stages_ls
        test_results = np.ones((len(stages_ls),len(stages_ls)))+100
        for ia, stage_name_a in enumerate(stages_ls):
            stage_sel_a = stage_duration_spike_rate_df.Stage.str.fullmatch(stage_name_a, case=False)
            spike_rate_a = stage_duration_spike_rate_df.SpikeOccRate[stage_sel_a].to_numpy()
            assert stage_sel_a.sum() == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(stages_ls):
                stage_sel_b = stage_duration_spike_rate_df.Stage.str.fullmatch(stage_name_b, case=False)
                assert stage_sel_b.sum() == nr_pats, "More than one entry per patient"
                spike_rate_b = stage_duration_spike_rate_df.SpikeOccRate[stage_sel_b].to_numpy()
                assert len(spike_rate_a) == len(spike_rate_b), "Spike rates for different stages have different lengths"

                # run Wilcoxon signed-rank test
                if ia != ib:
                    _, p_val_a = stats.shapiro(spike_rate_a)
                    _, p_val_b = stats.shapiro(spike_rate_b)

                    # If both samples are normally distributed, use paired t-test, otherwise use Wilcoxon signed-rank test
                    #if p_val_a >= 0.05 and p_val_b >= 0.05:
                    if False:
                        # run paired t-test
                        t_stat, p_val = stats.ttest_rel(spike_rate_a, spike_rate_b, nan_policy='raise', alternative='two-sided')
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
                    else:
                        # run Wilcoxon signed-rank test
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                        # run Wilcoxon signed-rank test
                        alternative_str = 'greater'
                        if np.mean(spike_rate_a) < np.mean(spike_rate_b):
                            alternative_str = 'less'
                        alternative_str = 'two-sided'
                        wilcoxon_stat, p_val = stats.wilcoxon(spike_rate_a, spike_rate_b, nan_policy='raise', alternative=alternative_str)
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                    test_results[ia,ib] = p_val
        pass

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05

        correction_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        correction_methods = ['fdr_bh']

        for method in correction_methods:
            corrected_test_results = test_results.copy()
            for ri in range(corrected_test_results.shape[0]):
                _, corrected_p_values, _, _ = multipletests(test_results[ri][test_results[ri]<100], alpha=0.05, method=method) # bonferroni, sidak, holm-sidak, holm, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky
                corrected_test_results[ri][test_results[ri]<100] = corrected_p_values

            #print(f"Bonferroni corrected threshold: {threshold:.3f}")
            #print(f"Holm-Bonferroni corrected p-values:\n{corrected_test_results}")
            #print(f"Uncorrected p-values:\n{test_results}")

            # Plot the heatmap of the test results
            fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
            ax = sns.heatmap(corrected_test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 32}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
            cbar = ax.collections[0].colorbar
            # Adjust the font size of the colorbar tick labels
            cbar.ax.tick_params(labelsize=32) # Set specific font size
            cbar.set_label('p value', fontsize=32) # Set colorbar label

            ax.grid(False)
            ax.set_xticklabels(stages_ls, rotation=45, fontsize=32)
            ax.set_yticklabels(stages_ls, rotation=0, fontsize=32)
            alpha_str = r" $\alpha$"        
            plt.title(f"Spike Occurrence Rate\nWilcoxon Signed-Rank Test p-values ({method} corrected)\n({alpha_str}:{threshold})", fontsize=36)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.tight_layout()
            plt.savefig(self.images_output_path / f"Spike_OccRate_Wilcoxon_Test_Results_{method}_corrected.png")
            plt.close()
        pass

    def plot_spike_activity_stages_differences(self, spike_data_df:pd.DataFrame=None):

        plot_spike_data_df = spike_data_df.copy()
        # Convert Amplitude to uV
        plot_spike_data_df.loc[:, 'Amplitude'] = plot_spike_data_df.Amplitude.values*1000*1000 # convert to uV
        # Get average spike activity per stage for each patient
        all_pats_avg_stage_activity = plot_spike_data_df[['Patient', 'Stage', 'Amplitude']].groupby(['Patient','Stage']).mean().reset_index()

        nr_pats = len(all_pats_avg_stage_activity.Patient.unique())
        print("\n\nPlotting Spike Activity per Sleep Stage")
        print(f"Nr. Patients: {nr_pats}")

        # Analyze Spike Activity
        stages_ci_ranges = {}
        for stage_name in self.sleep_stages_ls:
            stage_sel = all_pats_avg_stage_activity.Stage.str.fullmatch(stage_name, case=False)
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            all_pats_activity = all_pats_avg_stage_activity.loc[stage_sel, 'Amplitude'].to_numpy()
            ci_range = self.bootstrap_confidence_interval_univariate(all_pats_activity, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = ci_range
            # test normality
            _, p_val = stats.shapiro(all_pats_activity)
            normality = "normal"
            if p_val < 0.05:
                normality = "not normal"
            print(f"{normality} -- {stage_name} ({np.mean(all_pats_activity):.2f}, CI={ci_range[0]:.2f}-{ci_range[1]:.2f}) (uV)")
            pass

        max_ci_limit = max([max(v) for k,v in stages_ci_ranges.items()])

        # Plot patient average spike activity by sleep stage
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        assert nr_pats == len(all_pats_avg_stage_activity.Patient.unique()), "More than one entry per patient"
        errorbar_def = ("ci", 95)  # Percentile interval for error bars
        errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
        bp_ax = sns.barplot(data=all_pats_avg_stage_activity, x='Stage', y='Amplitude', hue='Stage', 
            order=self.sleep_stages_ls, palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        
        max_bar_height = 0
        for cont in bp_ax.containers:
            plt.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')

        axs.set_ylabel("Spike Activity (uV)", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim(0, max_ci_limit*1.1)
        #axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_title(f"{self.study_name}", fontsize=48)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Average_Spike_Activity.png")
        #plt.waitforbuttonpress()
        plt.close()

    def analyze_spike_activity_stages_differences(self, spike_data_df:pd.DataFrame=None):
        patients_ls = list(spike_data_df.Patient.unique())
        nr_pats = len(patients_ls)
        print(f"Analyzing Spike Activity differences between Sleep Stages")
        print(f"Nr. Patients: {nr_pats}")

        # Get average spike activity per stage for each patient
        all_pats_avg_stage_activity = spike_data_df[['Patient', 'Stage', 'Amplitude']].groupby(['Patient','Stage']).mean().reset_index()

        stages_ls = self.sleep_stages_ls
        test_results = np.ones((len(stages_ls),len(stages_ls)))+100
        for ia, stage_name_a in enumerate(stages_ls):
            stage_sel_a = all_pats_avg_stage_activity.Stage.str.fullmatch(stage_name_a, case=False)
            spike_activity_a = all_pats_avg_stage_activity.Amplitude[stage_sel_a].to_numpy()
            assert stage_sel_a.sum() == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(stages_ls):
                stage_sel_b = all_pats_avg_stage_activity.Stage.str.fullmatch(stage_name_b, case=False)
                assert stage_sel_b.sum() == nr_pats, "More than one entry per patient"
                spike_activity_b = all_pats_avg_stage_activity.Amplitude[stage_sel_b].to_numpy()
                assert len(spike_activity_a) == len(spike_activity_b), "Spike rates for different stages have different lengths"

                # run Wilcoxon signed-rank test
                if ia != ib:
                    _, p_val_a = stats.shapiro(spike_activity_a)
                    _, p_val_b = stats.shapiro(spike_activity_b)

                    # If both samples are normally distributed, use paired t-test, otherwise use Wilcoxon signed-rank test
                    #if p_val_a >= 0.05 and p_val_b >= 0.05:
                    if False:
                        # run paired t-test
                        t_stat, p_val = stats.ttest_rel(spike_activity_a, spike_activity_b, nan_policy='raise', alternative='two-sided')
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
                    else:
                        # run Wilcoxon signed-rank test
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                        # run Wilcoxon signed-rank test
                        alternative_str = 'greater'
                        if np.mean(spike_activity_a) < np.mean(spike_activity_b):
                            alternative_str = 'less'
                        alternative_str = 'two-sided'
                        wilcoxon_stat, p_val = stats.wilcoxon(spike_activity_a, spike_activity_b, nan_policy='raise', alternative=alternative_str)
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                    test_results[ia,ib] = p_val
        pass

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05

        correction_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        correction_methods = ['fdr_bh']

        for method in correction_methods:
            corrected_test_results = test_results.copy()
            for ri in range(corrected_test_results.shape[0]):
                _, corrected_p_values, _, _ = multipletests(test_results[ri][test_results[ri]<100], alpha=0.05, method=method) # bonferroni, sidak, holm-sidak, holm, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky
                corrected_test_results[ri][test_results[ri]<100] = corrected_p_values

            #print(f"Bonferroni corrected threshold: {threshold:.3f}")
            #print(f"Holm-Bonferroni corrected p-values:\n{corrected_test_results}")
            #print(f"Uncorrected p-values:\n{test_results}")

            # Plot the heatmap of the test results
            fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
            ax = sns.heatmap(corrected_test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 32}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
            cbar = ax.collections[0].colorbar
            # Adjust the font size of the colorbar tick labels
            cbar.ax.tick_params(labelsize=32) # Set specific font size
            cbar.set_label('p value', fontsize=32) # Set colorbar label

            ax.grid(False)
            ax.set_xticklabels(stages_ls, rotation=45, fontsize=32)
            ax.set_yticklabels(stages_ls, rotation=0, fontsize=32)
            alpha_str = r" $\alpha$"
            plt.title(f"Spike Activity\nWilcoxon Signed-Rank Test p-values ({method} corrected)\n({alpha_str}:{threshold})", fontsize=36)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.tight_layout()
            plt.savefig(self.images_output_path / f"Spike_Activity_Wilcoxon_Test_Results_{method}_corrected.png")
            plt.close()

    def plot_soz_vs_nonsoz_activity(self, spike_data_df:pd.DataFrame=None):
        # Plot SOZ vs Non-SOZ differences in spike activity across sleep stages
        patients_ls = list(spike_data_df.Patient.unique())
        nr_pats = len(patients_ls)
        print(f"Plotting SOZ vs Non-SOZ spike activity in each sleep stages")
        print(f"Nr. Patients: {nr_pats}")

        # Convert Amplitude to uV
        spike_data_df.loc[:, 'Amplitude'] = spike_data_df.Amplitude.values*1000*1000

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, len(self.sleep_stages_ls), figsize=FIGSIZE)

        # Compare SOZ vs Non-SOZ in the different sleep stages
        for si, stage_name in enumerate(self.sleep_stages_ls):
            plt_ax = axs[si]
            stage_data_df = spike_data_df[spike_data_df.Stage.str.fullmatch(stage_name, case=False)].copy().reset_index(drop=True)
                        
            colors_soz = [c for c in self.stages_colors[stage_name]]
            colors_soz.append(1)  # Add alpha channel for transparency
            colors_non_soz = [c for c in self.stages_colors[stage_name]]
            colors_non_soz.append(0.5)  # Add alpha channel for transparency
            soz_colors_dict = {'SOZ':colors_soz, 'Non-SOZ':colors_non_soz}

            # plot barplot with error bars
            errorbar_def = ("ci", 95)  # Percentile interval for error bars
            errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
            bp_ax = sns.barplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', palette=soz_colors_dict, 
                order=['Non-SOZ', 'SOZ'],
                capsize=.2,
                errorbar=errorbar_def,
                err_kws=errorbar_characteristics,
                linewidth=1, edgecolor=".5", 
                width=0.9, gap=0.00,
                estimator=np.mean,
                ax=plt_ax
                )
            # Add bar labels
            for cont in bp_ax.containers:
                labels = plt_ax.bar_label(cont, fmt='%.2f', fontsize=32, label_type='center', padding=0, rotation=90, color='black', weight='bold')
                # Adjust label x position
                for label in labels:
                    x, y = label.get_position()
                    label.set_x(x + 5)  # Adjust 
            
            # Add hatch pattern to differentiate SOZ and Non-SOZ
            for pi, patch in enumerate(plt_ax.patches):
                if pi == 0:
                    # r, g, b, a = patch.get_facecolor() # Get current color (including alpha)
                    #patch.set_facecolor((r, g, b, 0.5))
                    patch.set_hatch('..')  # Add hatch pattern to differentiate SOZ and Non-SOZ
                    fc = patch.get_facecolor()
                    patch.set_edgecolor(fc)
                    patch.set_facecolor('none')

            # Show y-axis label only for the first subplot
            if si == 0:
                plt_ax.set_ylabel("Spike Activity (uV)", fontsize=32)
            else:
                plt_ax.set_ylabel("")
                plt_ax.set_yticklabels("")

            # Plot legend, set the handles and labels manually
            # This is necessary to avoid grabbing the legend from the barplot's error bars
            # and to ensure the legend is only for SOZ and Non-SOZ
            handles = []
            labels = []
            for bar, label in zip(bp_ax.patches[:2], ['Non-SOZ', 'SOZ']):  # Adjust [:2] if you have more bars
                handles.append(bar)
                labels.append(label)
            plt_ax.legend(handles, labels, fontsize=20, loc='upper left', frameon=False)

            plt_ax.set_xlabel(stage_name, fontsize=32)
            # plt_ax.tick_params(axis='x', labelsize=32, rotation=60)
            plt_ax.set_xticklabels("")
            plt_ax.tick_params(axis='y', labelsize=32)

            # if 'Freiburg' in self.study_name:
            #     plt_ax.set_ylim(0, 180)
            # else:
            #     plt_ax.set_ylim(0, 310)
            plt_ax.set_ylim(0, 180)
            plt_ax.set_title(f"{stage_name}", fontsize=32, color=self.stages_colors[stage_name], weight='bold')
            plt_ax.set_title('')
            # plt.xticks(fontsize=32)
            # plt.yticks(fontsize=32)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}", fontsize=48)
        plt.subplots_adjust(wspace=3)
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Spike_Activity_SOZ_vs_NonSOZ_Hypothesis_Tests.png")
        #plt.waitforbuttonpress()
        plt.close()
        #print(res_dict)


        # Perform Mann-Whitney U test for SOZ vs Non-SOZ differences in spike activity across sleep stages
        res_dict = {'Stage':[], 'U_statistic':[], 'p_value':[]}
        stages_ci_ranges = {}
        print(f"\n\nPerforming Mann-Whitney U test for SOZ vs Non-SOZ differences in spike activity across sleep stages")
        print("Stage\tNon-SOZ Mean\tNon-SOZ CI\tSOZ Mean\tSOZ CI\tp-value")
        nr_soz = 0
        nr_non_soz = 0
        for si, stage_name in enumerate(self.sleep_stages_ls):
            plt_ax = axs[si]
            stage_data_df = spike_data_df[spike_data_df.Stage.str.fullmatch(stage_name, case=False)].copy().reset_index(drop=True)
            stage_data_df.loc[:, 'Amplitude'] = stage_data_df.Amplitude.values

            spike_data_df.Stage.str.fullmatch(stage_name, case=False)
            soz_amplitudes = stage_data_df.Amplitude[stage_data_df.SOZ.str.fullmatch('SOZ', case=False)].values
            non_soz_amplitudes = stage_data_df.Amplitude[stage_data_df.SOZ.str.fullmatch('Non-SOZ', case=False)].values
            res = stats.mannwhitneyu(soz_amplitudes, non_soz_amplitudes)
            nr_soz = len(soz_amplitudes)
            nr_non_soz = len(non_soz_amplitudes)
            res_dict['Stage'].append(stage_name)
            res_dict['U_statistic'].append(res.statistic)
            res_dict['p_value'].append(res.pvalue)

            ci_range_soz = self.bootstrap_confidence_interval_univariate(soz_amplitudes, func=np.mean, confidence_level=0.95, n_resamples=10000)
            ci_range_nonsoz = self.bootstrap_confidence_interval_univariate(non_soz_amplitudes, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = {'SOZ': ci_range_soz, 'Non-SOZ': ci_range_nonsoz}
            # test normality
            _, p_val_soz = stats.shapiro(soz_amplitudes)
            _, p_val_non_soz = stats.shapiro(non_soz_amplitudes)
            normality_soz = "normal"
            normality_non_soz = "normal"
            if p_val_soz < 0.05:
                normality_soz = "not normal"
            if p_val_non_soz < 0.05:
                normality_non_soz = "not normal"

            #print(f"\nMann-Whitney U test for {stage_name} SOZ vs Non-SOZ, p={res.pvalue:.3e} (U={res.statistic:.2f}, n_SOZ={nr_soz}, n_Non-SOZ={nr_non_soz})")
            #print(f"{normality_soz} -- {stage_name} SOZ= {np.mean(soz_amplitudes):.2f} (CI={ci_range_soz[0]:.2f}-{ci_range_soz[1]:.2f}) (uV)")
            #print(f"{normality_non_soz} -- {stage_name} Non-SOZ= {np.mean(non_soz_amplitudes):.2f} (CI={ci_range_nonsoz[0]:.2f}-{ci_range_nonsoz[1]:.2f}) (uV)")
            print(f"{stage_name}\t{np.mean(non_soz_amplitudes):.2f}\t{ci_range_nonsoz[0]:.2f}-{ci_range_nonsoz[1]:.2f}\t{np.mean(soz_amplitudes):.2f}\t{ci_range_soz[0]:.2f}-{ci_range_soz[1]:.2f}\t{res.pvalue:.3e}")
        
        print(f"\nNr. SOZ: {nr_soz}, Nr. Non-SOZ: {nr_non_soz}")
        return pd.DataFrame(res_dict)

    def plot_soz_vs_nonsoz_diff(self, spike_data_df:pd.DataFrame=None):
        # Analyze SOZ vs Non-SOZ differences in spike activity across sleep stages
        patients_ls = list(spike_data_df.Patient.unique())
        nr_pats = len(patients_ls)
        print(f"Analyzing SOZ vs Non-SOZ differences in spike activity across sleep stages")
        print(f"Nr. Patients: {nr_pats}")

        # Convert Amplitude to uV
        spike_data_df.loc[:, 'Amplitude'] = spike_data_df.Amplitude.values*1000*1000 # convert to uV

        # Get average spike activity per stage for each patient
        all_pats_avg_stage_activity = spike_data_df[['Patient', 'Stage', 'SOZ', 'Amplitude']].groupby(['Patient','Stage','SOZ']).mean().reset_index()

        # Get avg SOZ vs Non-SOZ differences in spike activity
        all_pats_diff = {}
        for stage_name in self.sleep_stages_ls:
            soz_avg_activity = all_pats_avg_stage_activity['Amplitude'][(all_pats_avg_stage_activity['SOZ']=='SOZ')&(all_pats_avg_stage_activity['Stage']==stage_name)].values
            nsoz_avg_activity = all_pats_avg_stage_activity['Amplitude'][(all_pats_avg_stage_activity['SOZ']=='Non-SOZ')&(all_pats_avg_stage_activity['Stage']==stage_name)].values

            assert soz_avg_activity.shape[0] == nr_pats, f"More than one entry per patient for SOZ in stage {stage_name}"
            assert nsoz_avg_activity.shape[0] == nr_pats, f"More than one entry per patient for Non-SOZ in stage {stage_name}"

            all_pats_diff[stage_name] = soz_avg_activity-nsoz_avg_activity
            pass
        
        # Analyze Activity Differences
        stages_ci_ranges = {}
        for stage_name in self.sleep_stages_ls:
            activity_diff = all_pats_diff[stage_name]
            assert len(activity_diff) == nr_pats, "More than one entry per patient"
            # Calculate confidence interval
            ci_range = self.bootstrap_confidence_interval_univariate(activity_diff, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = ci_range
            # test normality
            _, p_val = stats.shapiro(activity_diff)
            normality = "normal"
            if p_val < 0.05:
                normality = "not normal"
            print(f"{normality} -- {stage_name} ({np.mean(activity_diff):.2f}, CI={ci_range[0]:.2f}-{ci_range[1]:.2f}) (uV)")
            pass
        max_ci_limit = max([max(v) for k,v in stages_ci_ranges.items()])

        # Create DataFrame for plotting
        # Use melt to reshape the DataFrame for seaborn, what melt does is to transform the DataFrame from wide format to long format, 
        # i.e., it creates a new DataFrame with three columns: 'index', 'Stage', and 'Amplitude'
        all_pats_diff_df = pd.DataFrame(all_pats_diff).reset_index().melt(id_vars='index', var_name='Stage', value_name='Amplitude', ignore_index=True)

        # Plot Activity Differences
        print(f"Plotting SOZ vs Non-SOZ differences in spike activity across sleep stages")
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        errorbar_def = ("ci", 95)  # Percentile interval for error bars
        errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
        bp_ax = sns.barplot(data=all_pats_diff_df, x='Stage', y='Amplitude', hue='Stage',
            order=self.sleep_stages_ls, palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        for cont in bp_ax.containers:
            plt.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("Spikes / electrode / min.", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim(0, max_ci_limit*1.1)
        #axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_title(f"{self.study_name}", fontsize=48)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_NSOZ_Activity_Differences.png")
        #plt.waitforbuttonpress()
        plt.close()

    def analyze_soz_vs_nonsoz_diff(self, spike_data_df:pd.DataFrame=None):
        # Analyze SOZ vs Non-SOZ differences in spike activity across sleep stages
        patients_ls = list(spike_data_df.Patient.unique())
        nr_pats = len(patients_ls)
        print(f"Analyzing SOZ vs Non-SOZ differences in spike activity across sleep stages")
        print(f"Nr. Patients: {nr_pats}")

        # Convert Amplitude to uV
        spike_data_df.loc[:, 'Amplitude'] = spike_data_df.Amplitude.values*1000*1000 # convert to uV

        # Get average spike activity per stage for each patient
        all_pats_avg_stage_activity = spike_data_df[['Patient', 'Stage', 'SOZ', 'Amplitude']].groupby(['Patient','Stage','SOZ']).mean().reset_index()

        # Get avg SOZ vs Non-SOZ differences in spike activity
        all_pats_diff = {}
        for stage_name in self.sleep_stages_ls:
            soz_avg_activity = all_pats_avg_stage_activity['Amplitude'][(all_pats_avg_stage_activity['SOZ']=='SOZ')&(all_pats_avg_stage_activity['Stage']==stage_name)].values
            nsoz_avg_activity = all_pats_avg_stage_activity['Amplitude'][(all_pats_avg_stage_activity['SOZ']=='Non-SOZ')&(all_pats_avg_stage_activity['Stage']==stage_name)].values

            assert soz_avg_activity.shape[0] == nr_pats, f"More than one entry per patient for SOZ in stage {stage_name}"
            assert nsoz_avg_activity.shape[0] == nr_pats, f"More than one entry per patient for Non-SOZ in stage {stage_name}"

            all_pats_diff[stage_name] = soz_avg_activity-nsoz_avg_activity
            pass

        # Analyze Activity Differences        
        stages_ls = self.sleep_stages_ls
        test_results = np.ones((len(stages_ls),len(stages_ls)))+100
        for ia, stage_name_a in enumerate(stages_ls):
            spike_activity_a_diff = all_pats_diff[stage_name_a]
            assert len(spike_activity_a_diff) == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(stages_ls):
                spike_activity_b_diff = all_pats_diff[stage_name_b]
                assert len(spike_activity_b_diff) == nr_pats, "More than one entry per patient"

                # run Wilcoxon signed-rank test
                if ia != ib:
                    _, p_val_a = stats.shapiro(spike_activity_a_diff)
                    _, p_val_b = stats.shapiro(spike_activity_b_diff)

                    # If both samples are normally distributed, use paired t-test, otherwise use Wilcoxon signed-rank test
                    #if p_val_a >= 0.05 and p_val_b >= 0.05:
                    if False:
                        # run paired t-test
                        t_stat, p_val = stats.ttest_rel(spike_activity_a, spike_activity_b, nan_policy='raise', alternative='two-sided')
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
                    else:
                        # run Wilcoxon signed-rank test
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                        # run Wilcoxon signed-rank test
                        alternative_str = 'greater'
                        if np.mean(spike_activity_a_diff) < np.mean(spike_activity_b_diff):
                            alternative_str = 'less'
                        alternative_str = 'two-sided'
                        wilcoxon_stat, p_val = stats.wilcoxon(spike_activity_a_diff, spike_activity_b_diff, nan_policy='raise', alternative=alternative_str)
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                    test_results[ia,ib] = p_val
        pass

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05

        correction_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        correction_methods = ['fdr_bh']

        for method in correction_methods:
            corrected_test_results = test_results.copy()
            for ri in range(corrected_test_results.shape[0]):
                _, corrected_p_values, _, _ = multipletests(test_results[ri][test_results[ri]<100], alpha=0.05, method=method) # bonferroni, sidak, holm-sidak, holm, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky
                corrected_test_results[ri][test_results[ri]<100] = corrected_p_values

            #print(f"Bonferroni corrected threshold: {threshold:.3f}")
            #print(f"Holm-Bonferroni corrected p-values:\n{corrected_test_results}")
            #print(f"Uncorrected p-values:\n{test_results}")

            # Plot the heatmap of the test results
            fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
            ax = sns.heatmap(corrected_test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 32}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
            cbar = ax.collections[0].colorbar
            # Adjust the font size of the colorbar tick labels
            cbar.ax.tick_params(labelsize=32) # Set specific font size
            cbar.set_label('p value', fontsize=32) # Set colorbar label

            ax.grid(False)
            ax.set_xticklabels(stages_ls, rotation=45, fontsize=32)
            ax.set_yticklabels(stages_ls, rotation=0, fontsize=32)
            alpha_str = r" $\alpha$"
            plt.title(f"Spike Activity\nWilcoxon Signed-Rank Test p-values ({method} corrected)\n({alpha_str}:{threshold})", fontsize=36)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.tight_layout()
            plt.savefig(self.images_output_path / f"SOZ_vs_NSOZ_Spike_Activity_Difference_{method}_corrected.png")
            plt.close()

        pass


    def predict_soz_with_spike_activity(self, spike_data_df:pd.DataFrame=None, add_features:bool=False):
        
        nr_pats = len(spike_data_df.Patient.unique())

        print(f"Predicting SOZ with Spike Activity")
        print(f"Nr. Patients: {nr_pats}")

        # Predict SOZ based on Spike Activity
        prediction_results = {'TestPatID':[], 'TestPatient':[], 'Stage':[], 'Metric':[], 'Value':[], 'NrClinicalSzrs':[], 'NrElectroSzrs':[]}

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)

        for si, stage_name in enumerate(self.sleep_stages_ls):
            if stage_name == 'AllStages':
                stage_data_df = spike_data_df[spike_data_df.Stage!='Unknown']
            else:
                stage_data_df = spike_data_df[spike_data_df.Stage.str.fullmatch(stage_name, case=False)]
            pass

            roc_avg = {'fpr':np.linspace(0,1,1000), 'tpr':np.zeros_like(np.linspace(0,1,1000))}
            auroc_ls = []
            for pidx, pat_id in enumerate(stage_data_df.Patient.unique()):
                train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
                test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

                X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_train = train_set_df.SOZ.str.fullmatch('SOZ', case=False).to_numpy()
                X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
                
                X_test = test_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_test = test_set_df.SOZ.str.fullmatch('SOZ', case=False).to_numpy()

                assert np.unique(y_train).shape[0] > 1, "Train set has only one class"
                assert np.unique(y_test).shape[0] > 1, "Test set has only one class"

                # Basic feature engineering
                if add_features:
                    X_train = np.hstack([X_train, X_train**2, X_train**3])
                    X_test = np.hstack([X_test, X_test**2, X_test**3])

                model = LogisticRegression(penalty='l2', class_weight=None, solver='liblinear', max_iter=10000, tol=0.1)

                model.fit(X_train, y_train)
                y_predicted = model.predict(X_test)

                # Get probabilities for the positive class
                y_predicted_proba = model.predict_proba(X_test)[:, 1]
                # Calculate ROC Curve points
                fpr, tpr, thresholds = roc_curve(y_test, y_predicted_proba)
                # Calculate AUC
                y_predicted_bin = model.decision_function(X_test)

                auc_score = roc_auc_score(y_test, y_predicted_proba)  # Use a threshold of 0.5 for binary classification
                #auroc_val = roc_auc_score(y_test, y_predicted_bin)
                auroc_ls.append(auc_score)
                # print(f"Binary AUC Score: {auroc_val:.2f}")
                # print(f"Probab. AUC Score: {auc_score:.2f}")

                # interpolate tpr to have 100 points
                tpr_interp = np.interp(roc_avg['fpr'], fpr, tpr)

                roc_avg['tpr'] = np.add(roc_avg['tpr'], tpr_interp)
                
                pat_szrcnt_df = self.szr_cnt_df.loc[self.szr_cnt_df.PatID.str.fullmatch(pat_id, case=False)]

                mcc_val = matthews_corrcoef(y_test, y_predicted)
                prediction_results['TestPatient'].append(pidx)
                prediction_results['TestPatID'].append(pat_id)
                prediction_results['Stage'].append(stage_name)
                prediction_results['Metric'].append('AUROC')
                prediction_results['Value'].append(auc_score)
                prediction_results['NrClinicalSzrs'].append(pat_szrcnt_df.ClinicalSzr.values[0])
                prediction_results['NrElectroSzrs'].append(pat_szrcnt_df.SubclinicalSzr.values[0])

                prediction_results['TestPatient'].append(pidx)
                prediction_results['TestPatID'].append(pat_id)
                prediction_results['Stage'].append(stage_name)
                prediction_results['Metric'].append('MCC')
                prediction_results['Value'].append(mcc_val)
                prediction_results['NrClinicalSzrs'].append(pat_szrcnt_df.ClinicalSzr.values[0])
                prediction_results['NrElectroSzrs'].append(pat_szrcnt_df.SubclinicalSzr.values[0])
                pass

            roc_avg['tpr'] /= len(stage_data_df.Patient.unique())
            # = 1.0  # Ensure the last point is (1,1)
            prediction_results_df = pd.DataFrame(prediction_results)

            ci_range_auroc = self.bootstrap_confidence_interval_univariate(auroc_ls, func=np.mean, confidence_level=0.95, n_resamples=10000)

            label=f"{stage_name}({r"$\overline{\mathrm{AUROC}}$"}={np.mean(auroc_ls):.2f}, CI={ci_range_auroc[0]:.2f}-{ci_range_auroc[1]:.2f})"

            # Compare SOZ vs Non-SOZ
            axs.plot(roc_avg['fpr'], roc_avg['tpr'],color=self.stages_colors[stage_name], alpha=1, linewidth=8, linestyle='-',label=label)
            axs.plot(np.linspace(0,1,100), np.linspace(0,1,100),color='k', alpha=1, linewidth=1, linestyle='--')

            #axs.set_title(f"ROC, All Sleep Stages", fontsize=48)
            
            axs.set_ylabel("TPR", fontsize=32)
            axs.set_xlabel("FPR", fontsize=32)
            # if si==0:
            #     axs.set_ylabel("")
            #     axs.set_yticklabels("")

            axs.set_xticks(np.arange(0, 1.1, 0.1))
            axs.set_yticks(np.arange(0, 1.1, 0.1))
            axs.tick_params(axis='x', rotation=45, labelsize=32)          
            axs.tick_params(axis='y', rotation=0, labelsize=32)

            print(f"{stage_name} ({np.mean(auroc_ls):.2f}, CI={ci_range_auroc[0]:.2f}-{ci_range_auroc[1]:.2f})")       
        
        axs.set_title(f"SOZ Prediction\nROC Curves, All Sleep Stages", fontsize=48)
        axs.plot(np.linspace(0,1,100), np.linspace(0,1,100),color='k', alpha=1, linewidth=1, linestyle='--')    
        axs.legend(loc='lower right', fontsize=28, frameon=True, facecolor='w', edgecolor='k')
        axs.grid(True, linestyle='-', alpha=1, linewidth=1)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9, bottom=0.3, top=0.7)
        plt.savefig(self.images_output_path / "SOZ_Prediction_ROC_Curves.png")
        #plt.show()
        plt.close()
        
        return prediction_results_df
        
    def analyze_soz_prediction_performance(self, prediction_results_df:pd.DataFrame=None):
       
        # Analyze SOZ prediction performance
        nr_pats = len(prediction_results_df.TestPatient.unique())
        print(f"Analyzing SOZ prediction performance")
        print(f"Nr. Patients: {nr_pats}")

        # Get AUROC values for each stage
        stages_auroc = {}
        for stage_name in self.sleep_stages_ls:
            stage_data_df = prediction_results_df[prediction_results_df.Stage.str.fullmatch(stage_name, case=False)]
            auroc_vals = stage_data_df[stage_data_df.Metric=='AUROC'].Value.to_numpy()
            assert len(auroc_vals) == nr_pats, f"More than one entry per patient for AUROC in stage {stage_name}"
            stages_auroc[stage_name] = auroc_vals
            pass

        stat_val, p_val = stats.kruskal(stages_auroc['Sleep'], stages_auroc['N3'], stages_auroc['N2'], stages_auroc['N1'], stages_auroc['REM'], stages_auroc['Wake'])
        ########################
        stages_auroc_df = pd.DataFrame(stages_auroc).melt(var_name='Stage', value_name='AUROC', ignore_index=True)
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        errorbar_def = ("ci", 95)  # Percentile interval for error bars
        errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
        bp_ax = sns.barplot(data=stages_auroc_df, x='Stage', y='AUROC', hue='Stage',
            order=self.sleep_stages_ls, palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        for cont in bp_ax.containers:
            plt.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("AUROC", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylim(0, 1)
        #axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_title(f"{self.study_name}", fontsize=48)

        axs.grid(True, linestyle='-', alpha=1, linewidth=1)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / f"SOZ_prediction_Wilcoxon_Test_Results_Barchart.png")
        plt.close()
        
        ########################
        test_results = np.ones((len(self.sleep_stages_ls),len(self.sleep_stages_ls)))+100
        for ia, stage_name_a in enumerate( self.sleep_stages_ls):
            aurocs_a = stages_auroc[stage_name_a]
            assert len(aurocs_a) == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(self.sleep_stages_ls):
                aurocs_b = stages_auroc[stage_name_b]
                assert len(aurocs_b) == nr_pats, "More than one entry per patient"
                # run Wilcoxon signed-rank test
                if ia != ib:
                    _, p_val_a = stats.shapiro(aurocs_a)
                    _, p_val_b = stats.shapiro(aurocs_b)

                    # If both samples are normally distributed, use paired t-test, otherwise use Wilcoxon signed-rank test
                    #if p_val_a >= 0.05 and p_val_b >= 0.05:
                    if False:
                        # run paired t-test
                        t_stat, p_val = stats.ttest_rel(spike_activity_a, spike_activity_b, nan_policy='raise', alternative='two-sided')
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
                    else:
                        # run Wilcoxon signed-rank test
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                        # run Wilcoxon signed-rank test

                        alternative_str = 'greater'
                        if np.mean(aurocs_a) < np.mean(aurocs_b):
                            alternative_str = 'less'
                        alternative_str = 'two-sided'
                        wilcoxon_stat, p_val = stats.wilcoxon(aurocs_a, aurocs_b, nan_policy='raise', alternative=alternative_str) # two-sided, 'less', 'greater'
                        #print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                    test_results[ia,ib] = p_val
        pass

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05

        correction_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        correction_methods = ['fdr_bh']

        for method in correction_methods:
            corrected_test_results = test_results.copy()
            for ri in range(corrected_test_results.shape[0]):
                _, corrected_p_values, _, _ = multipletests(test_results[ri][test_results[ri]<100], alpha=0.05, method=method) # bonferroni, sidak, holm-sidak, holm, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky
                corrected_test_results[ri][test_results[ri]<100] = corrected_p_values

            #print(f"Bonferroni corrected threshold: {threshold:.3f}")
            #print(f"Holm-Bonferroni corrected p-values:\n{corrected_test_results}")
            #print(f"Uncorrected p-values:\n{test_results}")

            # Plot the heatmap of the test results
            fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
            ax = sns.heatmap(corrected_test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 32}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
            cbar = ax.collections[0].colorbar
            # Adjust the font size of the colorbar tick labels
            cbar.ax.tick_params(labelsize=32) # Set specific font size
            cbar.set_label('p value', fontsize=32) # Set colorbar label

            ax.grid(False)
            ax.set_xticklabels(self.sleep_stages_ls, rotation=45, fontsize=32)
            ax.set_yticklabels(self.sleep_stages_ls, rotation=0, fontsize=32)
            alpha_str = r" $\alpha$"
            plt.title(f"Spike Activity\nWilcoxon Signed-Rank Test p-values ({method} corrected)\n({alpha_str}:{threshold})", fontsize=36)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.tight_layout()
            plt.savefig(self.images_output_path / f"SOZ_prediction_Wilcoxon_Test_Results_{method}_corrected.png")
            plt.close()
            #######################

        return stages_auroc

    def plot_auroc_seizure_vs_no_seizure_patients(self, prediction_results_df:pd.DataFrame=None):

        prediction_results_df = prediction_results_df[prediction_results_df.Metric=='AUROC'].reset_index(drop=True).copy()

        # Plot SOZ vs Non-SOZ differences in spike activity across sleep stages
        patients_ls = list(prediction_results_df.TestPatID.unique())
        nr_pats = len(patients_ls)
        print(f"Plotting AUROC from patients with and without seizures")
        print(f"Nr. Patients: {nr_pats}")

        prediction_results_df['HasSeizures'] = np.logical_or(prediction_results_df.NrClinicalSzrs > 0, prediction_results_df.NrElectroSzrs > 0)
        prediction_results_df['HasSeizures'] = prediction_results_df.NrClinicalSzrs.to_numpy() > 0
        prediction_results_df['HasSeizures'] = prediction_results_df['HasSeizures'].map({True: 'HasSeizures', False: 'NoSeizures'})

        prediction_results_df = prediction_results_df.sort_values(by='HasSeizures').reset_index(drop=True)

        # Compare AUROC from patients with and without seizures
        fig, axs = plt.subplots(1, len(self.sleep_stages_ls), figsize=FIGSIZE)

        # Compare AUROC from the differenet sleep-stages and from patients with and without seizures
        legend_handles = []
        legend_labels = []
        aurocs_dict = {}
        for si, stage_name in enumerate(self.sleep_stages_ls):
            plt_ax = axs[si]
            stage_data_df = prediction_results_df[prediction_results_df.Stage.str.fullmatch(stage_name, case=False)].copy().reset_index(drop=True)
            auroc_no_seizures = stage_data_df.Value[stage_data_df.HasSeizures.str.fullmatch('NoSeizures', case=False)]
            auroc_has_seizures = stage_data_df.Value[stage_data_df.HasSeizures.str.fullmatch('HasSeizures', case=False)]
            assert auroc_no_seizures.shape[0] + auroc_has_seizures.shape[0] == nr_pats, f"More than one entry per patient for NoSeizures in stage {stage_name}"

            aurocs_dict[stage_name] = {'NoSeizures': auroc_no_seizures, 'HasSeizures': auroc_has_seizures}

            # Get confidence interval for AUROC
            ci_range_no_seizures = self.bootstrap_confidence_interval_univariate(auroc_no_seizures, func=np.mean, confidence_level=0.95, n_resamples=10000)
            ci_range_has_seizures = self.bootstrap_confidence_interval_univariate(auroc_has_seizures, func=np.mean, confidence_level=0.95, n_resamples=10000)
            print(f"\nStage: {stage_name}, NoSeizures mean AUROC={np.mean(auroc_no_seizures)}, CI= {ci_range_no_seizures[0]:.2f}-{ci_range_no_seizures[1]:.2f}")
            print(f"Stage: {stage_name}, HasSeizures mean AUROC={np.mean(auroc_has_seizures)}, CI= {ci_range_has_seizures[0]:.2f}-{ci_range_has_seizures[1]:.2f}")

            colors_soz = [c for c in self.stages_colors[stage_name]]
            colors_soz.append(1)  # Add alpha channel for transparency
            colors_non_soz = [c for c in self.stages_colors[stage_name]]
            colors_non_soz.append(0.5)  # Add alpha channel for transparency
            soz_colors_dict = {'HasSeizures':colors_soz, 'NoSeizures':colors_non_soz}

            # plot barplot with error bars
            errorbar_def = ("ci", 95)  # Percentile interval for error bars
            errorbar_characteristics = {'color': 'red', "linestyle":'-', "linewidth": 5, "alpha": 0.6}
            bp_ax = sns.barplot(data=stage_data_df, x='HasSeizures', y='Value', hue='HasSeizures', palette=soz_colors_dict, 
                order=['NoSeizures', 'HasSeizures'],
                capsize=.2,
                errorbar=errorbar_def,
                err_kws=errorbar_characteristics,
                linewidth=1, edgecolor=".5", 
                width=1, gap=0.0,
                estimator=np.mean,
                ax=plt_ax
                )
            # Add bar labels
            for cont in bp_ax.containers:
                labels = plt_ax.bar_label(cont, fmt='%.2f', fontsize=32, label_type='center', padding=0, rotation=90, color='black', weight='bold')
                # Adjust label x position
                for label in labels:
                    x, y = label.get_position()
                    label.set_x(x + 5)  # Adjust 
            
            # Add hatch pattern to differentiate SOZ and Non-SOZ
            for pi, patch in enumerate(plt_ax.patches):
                if pi == 0:
                    # r, g, b, a = patch.get_facecolor() # Get current color (including alpha)
                    #patch.set_facecolor((r, g, b, 0.5))
                    patch.set_hatch('..')  # Add hatch pattern to differentiate SOZ and Non-SOZ
                    fc = patch.get_facecolor()
                    patch.set_edgecolor(fc)
                    patch.set_facecolor('none')

            # Show y-axis label only for the first subplot
            if si == 0:
                plt_ax.set_ylabel("AUROC", fontsize=32)
            else:
                plt_ax.set_ylabel("")
                plt_ax.set_yticklabels("")

            # Plot legend, set the handles and labels manually
            # This is necessary to avoid grabbing the legend from the barplot's error bars
            # and to ensure the legend is only for SOZ and Non-SOZ
            for bar, patch_label in zip(bp_ax.patches[:2], ['NoSeizures', 'HasSeizures']):  # Adjust [:2] if you have more bars
                legend_handles.append(bar)
                legend_labels.append(patch_label)
            #plt_ax.legend(handles, labels, fontsize=20, loc='upper left', frameon=False)

            plt_ax.set_xlabel(stage_name, fontsize=32)
            # plt_ax.tick_params(axis='x', labelsize=32, rotation=60)
            plt_ax.set_xticklabels("")
            plt_ax.tick_params(axis='y', labelsize=32)

            plt_ax.set_ylim(0, 1)
            plt_ax.set_title(f"{stage_name}", fontsize=32, color=self.stages_colors[stage_name], weight='bold')
            plt_ax.set_title('')
            # plt.xticks(fontsize=32)
            # plt.yticks(fontsize=32)

            pass

        # Create new handles with black and white colors
        custom_handles = [Patch(facecolor='black', edgecolor='black', label=legend_labels[0]),
                          Patch(facecolor='none', edgecolor='black', label=legend_labels[1], hatch='..')]

        fig.legend(custom_handles, legend_labels[0:2], fontsize=32, loc='upper right', frameon=False)

        # Add a common title for the entire figure
        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}", fontsize=48)
        plt.subplots_adjust(wspace=1.1)
        #plt.tight_layout()
        plt.savefig(self.images_output_path / "AUROC_Score_HasSeizures_vs_NoSeizures.png")
        #plt.waitforbuttonpress()
        plt.close()
        #print(res_dict)


        # Perform Mann-Whitney U test
        print("\nPerforming Mann-Whitney U to test AUROC from patients with and without Seizures")
        for si, stage_name in enumerate(self.sleep_stages_ls):

            has_szrs_aurocs = aurocs_dict[stage_name]['HasSeizures']
            no_szrsaurocs = aurocs_dict[stage_name]['NoSeizures']

            # test normality
            _, p_val_soz = stats.shapiro(has_szrs_aurocs)
            _, p_val_non_soz = stats.shapiro(no_szrsaurocs)

            # Perform Mann-Whitney U test
            res = stats.mannwhitneyu(has_szrs_aurocs, no_szrsaurocs, alternative='two-sided')
            print(f"{stage_name} Mann-Whitney U test: U-statistic = {res.statistic:.2f}, p-value = {res.pvalue:.2e}")
            pass
             
        return
    

    def plot_soz_prediction_performance_vs_szr_count(self, prediction_results_df:pd.DataFrame=None):
        prediction_results_df = prediction_results_df[prediction_results_df.Metric=='AUROC'].reset_index(drop=True).copy()

        prediction_results_df['HasSeizures'] = prediction_results_df.NrClinicalSzrs>0
        #prediction_results_df['HasSeizures'] = prediction_results_df.NrElectroSzrs>0
        #prediction_results_df['HasSeizures'] = np.logical_or(prediction_results_df.NrClinicalSzrs.values>0, prediction_results_df.NrElectroSzrs.values>0)

        ###############
        # with_szrs_auroc = test_prediction_results_df.groupby('HasSeizures').Value.get_group('AllStages').values
        # stat_val, p_val = stats.kruskal(all_stages_auroc, n3_auroc, n2_auroc, n1_auroc, rem_auroc, wake_auroc)

        n2_szr_auroc = prediction_results_df[prediction_results_df.Stage=='N2'].groupby('HasSeizures').Value.get_group(True).values
        n3_szr_auroc = prediction_results_df[prediction_results_df.Stage=='N3'].groupby('HasSeizures').Value.get_group(True).values
        n1_szr_auroc = prediction_results_df[prediction_results_df.Stage=='N1'].groupby('HasSeizures').Value.get_group(True).values
        rem_szr_auroc = prediction_results_df[prediction_results_df.Stage=='REM'].groupby('HasSeizures').Value.get_group(True).values
        wake_szr_auroc = prediction_results_df[prediction_results_df.Stage=='Wake'].groupby('HasSeizures').Value.get_group(True).values
        #all_stages_szr_auroc = prediction_results_df[prediction_results_df.Stage=='AllStages'].groupby('HasSeizures').Value.get_group(True).values

        n2_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N2'].groupby('HasSeizures').Value.get_group(False).values
        n3_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N3'].groupby('HasSeizures').Value.get_group(False).values
        n1_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N1'].groupby('HasSeizures').Value.get_group(False).values
        rem_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='REM'].groupby('HasSeizures').Value.get_group(False).values
        wake_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='Wake'].groupby('HasSeizures').Value.get_group(False).values
        #all_stages_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='AllStages'].groupby('HasSeizures').Value.get_group(False).values

        mu_u, mu_p = stats.mannwhitneyu(n2_szr_auroc, n2_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"N2 AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")
        mu_u, mu_p = stats.mannwhitneyu(n3_szr_auroc, n3_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"N3 AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")
        mu_u, mu_p = stats.mannwhitneyu(n1_szr_auroc, n1_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"N1 AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")
        mu_u, mu_p = stats.mannwhitneyu(rem_szr_auroc, rem_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"REM AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")
        mu_u, mu_p = stats.mannwhitneyu(wake_szr_auroc, wake_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"Wake AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")
        # mu_u, mu_p = stats.mannwhitneyu(all_stages_szr_auroc, all_stages_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        # print(f"All Stages AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")

        pass
        ###############

        ###############

        analysis_stages = copy.copy(self.sleep_stages_ls)
        analysis_stages_colors = copy.copy(self.stages_colors)
        analysis_stages.insert(0, 'AllStages')
        analysis_stages_colors['AllStages'] = (0.5, 0.5, 0.5)

        nr_pats = len(prediction_results_df.TestPatID.unique())
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        sns.boxplot(data=prediction_results_df, x='HasSeizures', y='Value', hue='Stage', palette=analysis_stages_colors, ax=axs)
        axs.set_title(f"{self.study_name}\nSOZ Prediction Performance vs. Nr. Clinical Seizures\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_ylabel("AUROC", fontsize=32)
        axs.set_xlabel("Nr. Clinical Seizures", fontsize=32)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_Prediction_Performance_vs_SZR_Count.png")
        #plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":

    #pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Slow")
    pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Accelerated_MeanChFeats")
    #pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Accelerated_MedianChFeats")
    #pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Accelerated_SelectedChannelFeats")

    # Predict SOZ based on Spike Activity
    studies_ls =  [fr_ILAES2025_patients(), ACH_Pediatric_Patients_All()]
    studies_ls =  [ACH_Pediatric_Patients_All()]

    prediction_results_ls = []
    for study in studies_ls:
        print(f"Study: {study.dataset_name}")
        szr_cnt_info_fpath = Path("F:/Pediatric_Patients_Simultaneous/iEEG_Seizure_Info") / "Number_of_Seizures_Info.xlsx"
        if 'freiburg' in str(study.eeg_data_path).lower():
            images_opath = Path(os.getcwd()) / "Images_Output"
            szr_cnt_df = pd.read_excel(szr_cnt_info_fpath, sheet_name='Freiburg', engine='openpyxl')
        else:
            images_opath = Path(os.getcwd()) / "Images_Output_Pediatric"
            szr_cnt_df = pd.read_excel(szr_cnt_info_fpath, sheet_name='ACH', engine='openpyxl')
        os.makedirs(images_opath, exist_ok=True)

        pats_ls = list(study.patients.keys())
        characterization_datapath = pyeeg_output_path / "Spike_Characterized_Channels"
        stages_spikes_duration_rate_datapath = pyeeg_output_path / "Stage_Spike_Occurrence_Rate"
        an_sleep_stages_ls = ['Sleep', 'N3', 'N2', 'N1', 'REM', 'Wake']
        stages_colors = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}
        stages_colors = STAGES_COLORS
        for k,v in stages_colors.items():
            stages_colors[k] = (v[0]/255, v[1]/255, v[2]/255)
        spike_analyzer = Spike_Activity_Analyzer(study.dataset_name, characterization_datapath, stages_spikes_duration_rate_datapath, pats_ls, an_sleep_stages_ls, stages_colors, images_opath, szr_cnt_df)

        # Read and analyze sleep stages and spike occurrence rate data
        stage_duration_spike_rate_df = spike_analyzer.read_stages_duration_and_spike_rates()
        # Read and analyze spike actvity data (average spike amplitude)
        spike_data_df = spike_analyzer.read_patient_spike_data(stage_duration_spike_rate_df.copy())

        # Analyze differences in sleep stage durations
        spike_analyzer.plot_group_sleep_stage_durations_barchart(stage_duration_spike_rate_df)
        #spike_analyzer.plot_individual_sleep_stage_durations(stage_duration_spike_rate_df)
                
        # Analyze differences in spike occurrence rate between sleep stages
        spike_analyzer.plot_spike_occ_rate(stage_duration_spike_rate_df)
        spike_analyzer.analyze_sor_stages_differences(stage_duration_spike_rate_df)

        # Analyze differences in spike activity between wake and sleep stages    
        spike_analyzer.plot_spike_activity_stages_differences(spike_data_df.copy())
        spike_analyzer.analyze_spike_activity_stages_differences(spike_data_df.copy())
        
        # Localize SOZ based on spike activity
        spike_data_df = spike_analyzer.handle_patient_outliers(spike_data_df.copy())

        # Analyze SOZ vs Non-SOZ differences in spike activity
        spike_analyzer.plot_soz_vs_nonsoz_activity(spike_data_df.copy())
        #spike_analyzer.plot_soz_vs_nonsoz_diff(spike_data_df.copy())
        #spike_analyzer.analyze_soz_vs_nonsoz_diff(spike_data_df.copy())

        spike_data_df = spike_analyzer.get_patient_scaled_spike_data(spike_data_df)
        prediction_results_df = spike_analyzer.predict_soz_with_spike_activity(spike_data_df, add_features=False)
        spike_analyzer.analyze_soz_prediction_performance(prediction_results_df.copy())
        spike_analyzer.plot_auroc_seizure_vs_no_seizure_patients(prediction_results_df.copy())
        #spike_analyzer.plot_soz_prediction_performance_vs_szr_count(prediction_results_df)
        prediction_results_ls.append(prediction_results_df)
        pass

    #sys.exit()

    for i, study in enumerate(studies_ls):
        print(f"\n\nStudy: {study.dataset_name}")
        prediction_results_df = prediction_results_ls[i]      
        # print('Median AUROC')
        # median_auroc = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median()
        # iqr_auroc = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).quantile(0.75) - prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).quantile(0.25)
        # all_pats_auroc = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage'])
        # stages_names = median_auroc.index.tolist()
        # print("Stage\t Median AUROC\t IQR")
        # for stage_name in stages_names:
        #     print(f"{stage_name}\t {median_auroc.loc[stage_name].values[0]:.2f}\t\t {iqr_auroc.loc[stage_name].values[0]:.2f}")

        print(f"AUROC Results")
        nr_pats = len(list(study.patients.keys()))
        stages_ci_ranges = {}
        for stage_name in prediction_results_df.Stage.unique():
            stage_sel = np.logical_and(prediction_results_df.Stage.str.fullmatch(stage_name, case=False), prediction_results_df.Metric=='AUROC')
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            all_pats_auroc = prediction_results_df.loc[stage_sel, 'Value'].to_numpy()
            ci_range = Spike_Activity_Analyzer().bootstrap_confidence_interval_univariate(all_pats_auroc, func=np.mean, confidence_level=0.95, n_resamples=10000)
            stages_ci_ranges[stage_name] = ci_range
            print(f"{stage_name}: {np.mean(all_pats_auroc):.2f}, CI={ci_range[0]:.2f}-{ci_range[1]:.2f}")
            pass
        pass

        # print('Std_AUROC')
        # print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).std())

        # print('Median MCC')
        # print(prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).median())
        # print('Std_MCC')
        # print(prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).std())

