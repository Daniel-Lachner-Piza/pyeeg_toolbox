import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
import scikit_posthocs as sp
#import statsmodels as sm

from PIL import Image
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from studies_info import fr_ILAES2025_patients, ACH_Pediatric_Patients_All
from statsmodels.stats.anova import AnovaRM 

from imblearn.over_sampling import RandomOverSampler, SMOTE

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")

FIGSIZE = (16, 8)

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
        

    def read_patient_spike_data(self):
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
                stage_duration_spike_rate_df = pd.concat([stage_duration_spike_rate_df, pdata_df])
            except:
                print(f"File {pdata_fn} not found")

        stage_duration_spike_rate_df.reset_index(drop=True, inplace=True)

        # ANalyze the data
        stages_ls  = stage_duration_spike_rate_df.Stage.unique().tolist()
        data_analysis_dict = {
            'Stage': [],
            'Avg': [],
            'StdDev': [],
        }
        print('\nStageDurHours (Avg, StdDev)')
        for stage_name in stages_ls:
            stage_sel = stage_duration_spike_rate_df.Stage==stage_name
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            data_analysis_dict['Stage'].append(stage_name)
            data_analysis_dict['Avg'].append(stage_duration_spike_rate_df.StageDurM[stage_sel].mean()/60)
            data_analysis_dict['StdDev'].append(stage_duration_spike_rate_df.StageDurM[stage_sel].std()/60)
            print(f"{stage_name}: {data_analysis_dict['Avg'][-1]:.2f} (std.dev.: {data_analysis_dict['StdDev'][-1]:.2f}) hours")
            pass

        return stage_duration_spike_rate_df

    def handle_patient_outliers(self, spike_data_df:pd.DataFrame=None):

        # Remove outliers from the data
        raise_min = np.abs(spike_data_df.Amplitude.min())+0.1 # add 1 to avoid log(0)
        spike_data_df.loc[:, 'Amplitude'] = np.log(spike_data_df.Amplitude.values+raise_min) # log transform to reduce skewness, add 1 to avoid log(0)

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
        for pdata_fn in pats_ls:
            pdata_df = spike_data_df[spike_data_df.Patient.str.fullmatch(pdata_fn, case=False)].reset_index(drop=True).copy()
            pdata_df.Amplitude = MinMaxScaler().fit_transform(pdata_df.Amplitude.values.reshape(-1, 1)) # MinMaxScaler, StandardScaler()
            scaled_spike_data_df = pd.concat([scaled_spike_data_df, pdata_df])

        return scaled_spike_data_df


    def hypothesis_test_soz_vs_nonsoz(self, spike_data_df:pd.DataFrame=None):

        nr_pats = len(spike_data_df.Patient.unique())

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, 6, figsize=FIGSIZE)

        res = stats.mannwhitneyu(spike_data_df.Amplitude[spike_data_df.SOZ=='SOZ'], spike_data_df.Amplitude[spike_data_df.SOZ=='Non-SOZ'])
        nr_soz = len(spike_data_df[spike_data_df.SOZ=='SOZ'])
        nr_non_soz = len(spike_data_df[spike_data_df.SOZ=='Non-SOZ'])
        # All Stages
        plt_ax = axs[0]
        sns.boxplot(data=spike_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=axs[0])
        plt_ax.set_title(f"All Stages\n p-value = {res.pvalue:.2e}\n nr.SOZ={nr_soz}, nr.NonSOZ={nr_non_soz}")
        plt_ax.set_ylabel("Spike Activity \n(Scaled per Patient)")
        plt_ax.grid(color='0.8', linestyle='-', linewidth=0.5)

        # Compare SOZ vs Non-SOZ in the different sleep stages
        for si, stage_name in enumerate(self.sleep_stages_ls):
            plt_ax = axs[si+1]
            stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]
            res = stats.mannwhitneyu(stage_data_df.Amplitude[stage_data_df.SOZ=='SOZ'], stage_data_df.Amplitude[stage_data_df.SOZ=='Non-SOZ'])
            nr_soz = len(stage_data_df[stage_data_df.SOZ=='SOZ'])
            nr_non_soz = len(stage_data_df[stage_data_df.SOZ=='Non-SOZ'])
            sns.boxplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=plt_ax)
            plt_ax.set_title(f"{stage_name}\n p-value = {res.pvalue:.2e}\n nr.SOZ={nr_soz}, nr.NonSOZ={nr_non_soz}")
            plt_ax.set_ylabel("Spike Activity \n(Scaled per Patient)")
            plt_ax.grid(color='0.8', linestyle='-', linewidth=0.5)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}\nSpike Activity in SOZ vs. Non-SOZ\nNr. Patients={nr_pats}")
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Spike_Activity_SOZ_vs_NonSOZ_Hypothesis_Tests.png")
        #plt.waitforbuttonpress()
        plt.close()

    def oversample_patients(self, train_set_df):
        os_train_set_df = pd.DataFrame()
        for pat_id in train_set_df.Patient.unique():
            pat_df = train_set_df[train_set_df.Patient==pat_id]
            soz_df = pat_df[pat_df.SOZ=='SOZ']
            non_soz_df = pat_df[pat_df.SOZ=='Non-SOZ']
            n_p_ratio = int(np.round(len(non_soz_df)/len(soz_df)))+1
            if n_p_ratio>=2:
                soz_df = pd.concat([soz_df]*n_p_ratio)
            os_train_set_df = pd.concat([os_train_set_df, soz_df, non_soz_df])
        os_train_set_df = os_train_set_df.sample(frac=1).reset_index(drop=True)
        return os_train_set_df

    def predict_soz_with_spike_activity(self, spike_data_df:pd.DataFrame=None, add_features:bool=False):
        
        nr_pats = len(spike_data_df.Patient.unique())

        # Predict SOZ based on Spike Activity
        analysis_stages = copy.copy(self.sleep_stages_ls)
        analysis_stages_colors = copy.copy(self.stages_colors)
        analysis_stages.insert(0, 'AllStages')
        analysis_stages_colors['AllStages'] = (0.5, 0.5, 0.5)
        prediction_results = {'TestPatID':[], 'TestPatient':[], 'Stage':[], 'Metric':[], 'Value':[], 'NrClinicalSzrs':[], 'NrElectroSzrs':[]}
        for si, stage_name in enumerate(analysis_stages):
            if stage_name == 'AllStages':
                stage_data_df = spike_data_df[spike_data_df.Stage!='Unknown']
            else:
                stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]

            pass

            for pidx, pat_id in enumerate(stage_data_df.Patient.unique()):
                train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
                test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

                # Oversample to have equal positives and negatives
                #train_set_df = self.oversample_patients(train_set_df)
                # print class imbalance
                #print(f"Patient {pat_id} - Train Set Class Ratio: {np.sum(train_set_df.SOZ =='SOZ')/len(train_set_df)*100:.2f}%")

                X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_train = train_set_df.SOZ.to_numpy()=='SOZ'
                X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
                print(f"Positives to Negatives Ratio:{np.sum(y_train)/len(y_train)*100}")
                
                X_test = test_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_test = test_set_df.SOZ.to_numpy()=='SOZ'

                # scaler = StandardScaler()
                # X_train = scaler.fit_transform(X_train)
                # X_test = scaler.transform(X_test)

                # if np.unique(y_test).shape[0] == 1:
                #     print(f"Patient {pat_id} has only one class in test set")
                #     continue


                assert np.unique(y_train).shape[0] > 1, "Train set has only one class"
                assert np.unique(y_test).shape[0] > 1, "Test set has only one class"

                # Basic feature engineering
                if add_features:
                    X_train = np.hstack([X_train, X_train**2, X_train**3])
                    X_test = np.hstack([X_test, X_test**2, X_test**3])

                model = LogisticRegression(penalty='l2', class_weight=None, solver='liblinear', max_iter=10000, tol=0.1)

                model.fit(X_train, y_train)
                y_predicted = model.predict(X_test)

                # model = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', n_jobs=-1)
                # model.fit(X_train, y_train)
                # y_predicted = model.predict(X_test)

                # model = SVC()
                # model.fit(X_train, y_train)
                # y_predicted = model.predict(X_test)

                #print("Training Set", len(y_test))

                pat_szrcnt_df = self.szr_cnt_df.loc[self.szr_cnt_df.PatID.str.fullmatch(pat_id, case=False)]

                auroc_val = roc_auc_score(y_test, y_predicted)
                mcc_val = matthews_corrcoef(y_test, y_predicted)
                prediction_results['TestPatient'].append(pidx)
                prediction_results['TestPatID'].append(pat_id)
                prediction_results['Stage'].append(stage_name)
                prediction_results['Metric'].append('AUROC')
                prediction_results['Value'].append(auroc_val)
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

        prediction_results_df = pd.DataFrame(prediction_results)

        for sleep_stage in analysis_stages:
            auroc_vals = prediction_results_df[np.logical_and(prediction_results_df.Metric=='AUROC', prediction_results_df.Stage==sleep_stage)].Value.to_numpy()
            q75, q25 = np.percentile(auroc_vals, [75 ,25])
            iqr = q75 - q25
            print(f"Stage: {sleep_stage}, AUROC_IQR: {iqr:.2f}")
            pass

        print('Median AUROC')
        print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median())
        print('Std_AUROC')
        print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).std())
        pass

    ################
        # Conduct the repeated measures ANOVA 
        test_prediction_results_df = prediction_results_df[prediction_results_df.Metric=='AUROC'].reset_index(drop=True).copy()
        anova_results = AnovaRM(data=test_prediction_results_df, subject='TestPatient', depvar='Value', within=['Stage']).fit()
        print(anova_results)
        p_val_rmanova = anova_results.anova_table['Pr > F'].mean()


    ###############
        all_stages_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('AllStages').values
        n3_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N3').values
        n2_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N2').values
        n1_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N1').values
        rem_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('REM').values
        wake_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('Wake').values

        stat_val, p_val = stats.kruskal(all_stages_auroc, n3_auroc, n2_auroc, n1_auroc, rem_auroc, wake_auroc)
        pass

    # Perform Friedman test
    #     all_pats_test_data = []
    #     for stage in prediction_results_df[prediction_results_df.Metric=='AUROC'].Stage.unique():
    #         test_data = prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage==stage)].Value.to_numpy()
    #         all_pats_test_data.append(test_data)
    #     assert np.unique([len(vec) for vec in all_pats_test_data]).shape[0]==1, "Not all patients have the same number of entries per stage"
    #     all_pats_test_data = np.array(all_pats_test_data).T
    #     prediction_results_df[(prediction_results_df.Metric=='AUROC')].reset_index
    #     friedman_stat, friedman_p_val = stats.friedmanchisquare(
    #         prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage=='N3')].Value.to_numpy(),
    #         prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage=='N2')].Value.to_numpy(),
    #         prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage=='N1')].Value.to_numpy(),
    #         prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage=='REM')].Value.to_numpy(),
    #         prediction_results_df[(prediction_results_df.Metric=='AUROC')&(prediction_results_df.Stage=='Wake')].Value.to_numpy(),
    #         nan_policy='raise',
    #         )
    #     print(f"Friedman test: statistic = {friedman_stat:.2f}, p-value = {friedman_p_val:.2e}")

    #     # Perform Dunn's test for multiple comparisons
    #     p_values = sp.posthoc_dunn(spike_data_df_plot, val_col = 'Amplitude', group_col='Stage', p_adjust='bonferroni', sort=True)
    #     print(p_values)
    # ###############

        fig, axs = plt.subplots(1, 1, figsize=(4,8), sharey=True)
        #ax = axs[0]
        ax = axs
        box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='AUROC'], x='Stage', y='Value', hue='Stage', palette=analysis_stages_colors, ax=ax)
        axs.set_title(f"Area under the ROC Curve\nNr. Patients={nr_pats}", fontsize=24)
        axs.set_ylabel("AUROC", fontsize=24)
        medians_df = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
        vertical_offset = prediction_results_df[prediction_results_df.Metric=='AUROC'].Value.median() * 0.05 # offset from median for display
        for xtick in box_plot.get_xticks():
            median_val = medians_df.Value[medians_df.Stage==analysis_stages[xtick]].to_numpy()[0]
            median_str = f"{median_val:.2f}"
            box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size=24,color='w',weight='semibold') # size=10
            pass

        box_plot.text(x=box_plot.get_xticks()[-1], y=0.95, s=f"Repeated Measures Anova p_val: {p_val_rmanova:.2f}", horizontalalignment='center',size=14,color='r',weight='bold')


        #ax = axs[1]
        # box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='MCC'], x='Stage', y='Value', hue='Stage', palette=analysis_stages_colors, ax=ax)
        # axs.set_title(f"Matthews Correlation Coefficient\nNr. Patients={nr_pats}")
        # axs.set_ylabel("MCC")
        # medians_df = prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
        # vertical_offset = prediction_results_df[prediction_results_df.Metric=='MCC'].Value.median() * 0.05 # offset from median for display
        # for xtick in box_plot.get_xticks():
        #     median_val = medians_df.Value[medians_df.Stage==analysis_stages[xtick]].to_numpy()[0]
        #     median_str = f"{median_val:.2f}"
        #     box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size=16,color='w',weight='semibold')
        #     pass

        plt.tick_params(axis='x', labelsize=16)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}\nPrediction of SOZ", fontsize=36)
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_Prediction.png")
        #plt.waitforbuttonpress()
        plt.close()

        return prediction_results_df


    def plot_per_stage_spike_activity(self, spike_data_df:pd.DataFrame=None):
        
        spike_data_df_plot = spike_data_df.copy() 
        spike_data_df_plot.loc[:, 'Amplitude'] = spike_data_df_plot.Amplitude.values*1000*1000 # convert to uV
        nr_pats = len(spike_data_df.Patient.unique())

        # Perform Kruskal-Wallis test
        kruskal_stat, kruskal_p_val = stats.kruskal(
            spike_data_df_plot.Amplitude[spike_data_df_plot.Stage=='N3'], 
            spike_data_df_plot.Amplitude[spike_data_df_plot.Stage=='N2'], 
            spike_data_df_plot.Amplitude[spike_data_df_plot.Stage=='N1'], 
            spike_data_df_plot.Amplitude[spike_data_df_plot.Stage=='REM'],
            spike_data_df_plot.Amplitude[spike_data_df_plot.Stage=='Wake'],
            nan_policy='raise',
            )
        print(f"Kruskal-Wallis test: statistic = {kruskal_stat:.2f}, p-value = {kruskal_p_val:.2e}")

        # Perform Dunn's test for multiple comparisons
        p_values = sp.posthoc_dunn(spike_data_df_plot, val_col = 'Amplitude', group_col='Stage', p_adjust='bonferroni', sort=True)
        print(p_values)

        box_plot = sns.boxplot(data=spike_data_df_plot, x='Stage', y='Amplitude', hue='Stage', palette=self.stages_colors, showfliers=False)
        plt.ylabel("Spike Activity (uV)", fontsize=16)
        plt.xlabel("Sleep Stage", fontsize=16)
        plt.title(f"{self.study_name}\nSpike Activity per Sleep Stage\nNr. Patients={nr_pats}", fontsize=20)
 
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Average_Spike_Activity.png")
        #plt.waitforbuttonpress()
        plt.close()

        pass

    def plot_group_sleep_stage_durations(self, stage_duration_spike_rate_df:pd.DataFrame=None):
        nr_pats = len(stage_duration_spike_rate_df.PatID.unique())

        #sleep_ref_img_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\pyeeg_toolbox\\persyst\\SleepStages_Reference.png")
        sleep_ref_img_path = os.getcwd()+'/pyeeg_toolbox/persyst/SleepStages_Reference.png'
        sleep_ref_img= Image.open(sleep_ref_img_path)
        rsz_ratio = 2
        img_rsz = (int(sleep_ref_img.size[0]/rsz_ratio), int(sleep_ref_img.size[1]/rsz_ratio))
        sleep_ref_img = sleep_ref_img.resize(img_rsz)

        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        
        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM']
        sum_stages_dur_mins = []
        for stage_name in to_plot_stage_names:
            stage_sel = stage_duration_spike_rate_df.Stage==stage_name
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            sum_stages_dur_mins.append(stage_duration_spike_rate_df.StageDurM[stage_sel].sum())
            pass

        to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]

        sum_stages_dur_perc = (np.array(sum_stages_dur_mins)/np.sum(sum_stages_dur_mins))*100
        wedgeprops = {"edgecolor" : "white", 'linewidth': 5, 'antialiased': True}
        
        patches, texts, pcts = axs.pie(x=sum_stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':24, 'color':"w", 'weight':'bold'}, startangle=-200)
        for i, patch in enumerate(patches):
            texts[i].set_color(patch.get_facecolor())
        axs.set_ylabel("Relative Duration of Sleep Stages (%)", fontsize=24)
        axs.set_title(f"{self.study_name}\nProportion of summed duration of Sleep Stages\nNr.Patients = {nr_pats}", fontsize=24, color='black')
        pass
        #plt.legend(loc='lower right', fontsize=24)

        # Overlay image on plot
        im_width, im_height = sleep_ref_img.size
        bbox = fig.get_window_extent() 
        fig.figimage(sleep_ref_img, xo=int(bbox.x1-im_width/2), yo=int(bbox.y1-im_height/2), zorder=3, alpha=.7, origin='upper')

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
                stage_sel = patient_data_df.Stage==stage_name
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
            #patches, texts, pcts = ax.pie(x=sum_stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':24, 'color':"w", 'weight':'bold'}, startangle=-200)
            #patches, texts, pcts = ax.pie(x=sum_stages_dur_perc, labels=new_to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':12, 'color':"w", 'weight':'bold'}, startangle=-200)
            patches, texts = ax.pie(sum_stages_dur_perc, labels=new_to_plot_stage_names, colors=to_plot_stages_colors, startangle=-200)
            for i, patch in enumerate(patches):
                texts[i].set_color(patch.get_facecolor())
            # ax.set_ylabel("Relative Duration of Sleep Stages (%)", fontsize=24)
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

        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        assert nr_pats == len(stage_duration_spike_rate_df.PatID.unique()), "More than one entry per patient"
        #sns.violinplot(data=stage_duration_spike_rate_df, x='Stage', y='SpikeOccRate', hue='Stage', palette=self.stages_colors, ax=axs)
        bp_ax = sns.barplot(data=stage_duration_spike_rate_df, x='Stage', y='SpikeOccRate', hue='Stage', palette=self.stages_colors, ax=axs,
            capsize=.2,
            errorbar='sd',
            err_kws={'color': 'k', "linestyle":'dashed', "linewidth": 2, "alpha": 0.6},
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1
            )
        
        for cont in bp_ax.containers:
            plt.bar_label(cont, fmt='%.2f', fontsize=24, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("Spikes Occ.Rate/min.", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=24)
        #axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=48)
        axs.set_title(f"{self.study_name}", fontsize=48)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Average_Spike_OccRate.png")
        #plt.waitforbuttonpress()
        plt.close()

        # Analyze Spike Occurrence Rate
        stages_ls  = stage_duration_spike_rate_df.Stage.unique().tolist()
        data_analysis_dict = {
            'Stage': [],
            'Avg': [],
            'StdDev': [],
        }
        print('\nSpikeOccRate (Avg, StdDev)')
        for stage_name in stages_ls:
            stage_sel = stage_duration_spike_rate_df.Stage==stage_name
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            data_analysis_dict['Stage'].append(stage_name)
            data_analysis_dict['Avg'].append(stage_duration_spike_rate_df.SpikeOccRate[stage_sel].mean())
            data_analysis_dict['StdDev'].append(stage_duration_spike_rate_df.SpikeOccRate[stage_sel].std())
            print(f"{stage_name}: {data_analysis_dict['Avg'][-1]:.2f} (std.dev.: {data_analysis_dict['StdDev'][-1]:.2f}) spikes/min.")
        pass


    def analyze_spike_occ_rate_wake_sleep(self, stage_duration_spike_rate_df:pd.DataFrame=None):
        patients_ls = list(stage_duration_spike_rate_df.PatID.unique())
        nr_pats = len(patients_ls)

        # Analyze Spike Occurrence Rate
        stages_ls  = stage_duration_spike_rate_df.Stage.unique().tolist()
        data_analysis_dict = {
            'Stage': [],
            'Avg': [],
            'StdDev': [],
        }
        print('\nSpikeOccRate (Avg, StdDev)')
        test_results = np.ones((len(stages_ls),len(stages_ls)))
        for ia, stage_name_a in enumerate(stages_ls):
            stage_sel_a = stage_duration_spike_rate_df.Stage==stage_name_a
            spike_rate_a = stage_duration_spike_rate_df.SpikeOccRate[stage_sel_a].to_numpy()
            assert stage_sel_a.sum() == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(stages_ls):
                stage_sel_b = stage_duration_spike_rate_df.Stage==stage_name_b
                assert stage_sel_b.sum() == nr_pats, "More than one entry per patient"
                spike_rate_b = stage_duration_spike_rate_df.SpikeOccRate[stage_sel_b].to_numpy()
                assert len(spike_rate_a) == len(spike_rate_b), "Spike rates for different stages have different lengths"

                # run Wilcoxon signed-rank test
                if ia != ib:
                    wilcoxon_stat, wilcoxon_p_val = stats.wilcoxon(spike_rate_a, spike_rate_b, nan_policy='raise', alternative='two-sided')
                    print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")
                    test_results[ia,ib] = wilcoxon_p_val
                    # run paired t-test
                    t_stat, p_val = stats.ttest_rel(spike_rate_a, spike_rate_b, nan_policy='raise', alternative='two-sided')
                    print(f"Spike Occ.Rate {stage_name_a} vs {stage_name_b}\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
                pass
        pass

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05 / ((len(stages_ls) * (len(stages_ls) - 1))/2)  # Bonferroni correction for multiple comparisons
        threshold = 0.05 / (len(stages_ls)-1)  # Bonferroni correction for multiple comparisons
        print(f"Bonferroni corrected threshold: {threshold:.3f}")
                             
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        ax = sns.heatmap(test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 24}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
        cbar = ax.collections[0].colorbar
        # Adjust the font size of the colorbar tick labels
        cbar.ax.tick_params(labelsize=24) # Set specific font size
        cbar.set_label('p value', fontsize=24) # Set colorbar label

        #ax = sns.heatmap(test_results, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
        ax.grid(False)
        ax.set_xticklabels(stages_ls, rotation=45, fontsize=32)
        ax.set_yticklabels(stages_ls, rotation=0, fontsize=32)
        alpha_str = r" $\alpha$"        
        plt.title(f"Spike Occurrence Rate\nWilcoxon Signed-Rank Test p-values\n({alpha_str}:{threshold})", fontsize=36)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Spike_OccRate_Wilcoxon_Test_Results.png")
        plt.close()


        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        # Plot Wake vs Sleep Spike Occurrence Rate using line plot with marker
        patient_occ_rates = {'Patient':[], 'PatIdx':[], 'StageName':[], 'OccRate':[]}
        for idx, patient in enumerate(patients_ls):
            patient_data_df = stage_duration_spike_rate_df[stage_duration_spike_rate_df.PatID.str.fullmatch(patient, case=False)].reset_index(drop=True).copy()
            
            wake_rate = patient_data_df[patient_data_df.Stage=='Wake'].SpikeOccRate.mean()
            sel_sleep = np.logical_or.reduce((
                patient_data_df.Stage.str.fullmatch('N1', case=False),
                patient_data_df.Stage.str.fullmatch('N2', case=False),
                patient_data_df.Stage.str.fullmatch('N3', case=False),
                patient_data_df.Stage.str.fullmatch('REM', case=False),
            ))
            sleep_rate = patient_data_df.loc[sel_sleep, 'SpikeOccRate'].mean()
            patient_occ_rates['Patient'].append(patient)
            patient_occ_rates['PatIdx'].append(idx)
            patient_occ_rates['StageName'].append('Sleep')
            patient_occ_rates['OccRate'].append(sleep_rate)
            
            for sleep_stage_name in self.sleep_stages_ls:
                patient_occ_rates['Patient'].append(patient)
                patient_occ_rates['PatIdx'].append(idx)
                patient_occ_rates['StageName'].append(sleep_stage_name)

                stage_rate = patient_data_df[patient_data_df.Stage==sleep_stage_name].SpikeOccRate.mean()
                patient_occ_rates['OccRate'].append(stage_rate)

            wake_rate = patient_data_df[patient_data_df.Stage=='Wake'].SpikeOccRate.mean()
            axs.plot(1, wake_rate, marker='o', color='blue', markersize=12)
            axs.plot(2, sleep_rate, marker='o', color='orange', markersize=12)
            axs.plot([1,2], [wake_rate, sleep_rate], '--k', alpha=0.5)
            pass
        
        patient_occ_rates_df = pd.DataFrame(patient_occ_rates)
    ################
        # # Perform Friedman test
        # friedman_stat, friedman_p_val = stats.friedmanchisquare(
        #     patient_occ_rates['N3'], 
        #     patient_occ_rates['N2'],
        #     patient_occ_rates['N1'],
        #     patient_occ_rates['REM'],
        #     patient_occ_rates['WakeRate'],
        #     nan_policy='raise',
        #     )
        # print(f"Friedman test: statistic = {friedman_stat:.2f}, p-value = {friedman_p_val:.2e}")

        # # Perform Dunn's test for multiple comparisons
        # p_values = sp.posthoc_dunn(spike_data_df_plot, val_col = 'Amplitude', group_col='Stage', p_adjust='bonferroni', sort=True)
        # print(p_values)
    ###############
        
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Wake'].to_numpy(),
                                         patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Sleep'].to_numpy(),
                                           nan_policy='raise', alternative='two-sided')
        print(f"Occ.Rate WAKE VS: SLEEP\nPaired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
        # Perform Wilcoxon signed-rank test
        wilcoxon_stat, wilcoxon_p_val = stats.wilcoxon(patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Wake'].to_numpy(),
                                                       patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Sleep'].to_numpy(),
                                                       nan_policy='raise', alternative='two-sided')
        print(f"Occ.Rate WAKE VS: SLEEP\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.3f}")

        # N3 vs N2
        occ_rates_a = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Sleep'].to_numpy()
        occ_rates_b = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Wake'].to_numpy()
        t_stat, p_val = stats.ttest_rel(occ_rates_a, occ_rates_b, nan_policy='raise', alternative='greater')
        print(f"Paired t-test Sleep vs Wake: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")
        # N3 vs N2
        occ_rates_a = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N3'].to_numpy()
        occ_rates_b = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N2'].to_numpy()
        t_stat, p_val = stats.ttest_rel(occ_rates_a, occ_rates_b, nan_policy='raise', alternative='greater')
        print(f"Paired t-test N3 vs N2: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")
        # N3 vs N1
        occ_rates_a = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N3'].to_numpy()
        occ_rates_b = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N1'].to_numpy()
        t_stat, p_val = stats.ttest_rel(occ_rates_a, occ_rates_b, nan_policy='raise', alternative='greater')
        print(f"Paired t-test N3 vs N1: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")
        # N3 vs REM
        occ_rates_a = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N3'].to_numpy()
        occ_rates_b = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='REM'].to_numpy()
        t_stat, p_val = stats.ttest_rel(occ_rates_a, occ_rates_b, nan_policy='raise', alternative='greater')
        print(f"Paired t-test N3 vs REM: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")
        # N3 vs Wake
        occ_rates_a = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='N3'].to_numpy()
        occ_rates_b = patient_occ_rates_df.OccRate[patient_occ_rates_df.StageName=='Wake'].to_numpy()
        t_stat, p_val = stats.ttest_rel(occ_rates_a, occ_rates_b, nan_policy='raise', alternative='greater')
        print(f"Paired t-test N3 vs Wake: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")

        
        htest_res_str = f"Paired t-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}\nWilcoxon signed-rank test: statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p_val:.2e}"
        axs.text(0.5, 0.95, htest_res_str, ha='center', va='top', transform=axs.transAxes, fontsize=16, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))
        
        axs.set_xticks([1, 2])
        axs.set_xticklabels(['Wake', 'Sleep'], fontsize=18)
        axs.set_ylabel("Spike Occ.Rate/min.", fontsize=24)
        axs.set_title(f"{self.study_name}\nSpike Occ.Rate/min.\nNr.Patients = {nr_pats}", fontsize=24)
        #axs.set_ylim(0, 1.5)
        axs.set_xlim(0.5, 2.5)
        axs.grid(color='0.8', linestyle='-', linewidth=0.5)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.images_output_path / "Wake_Sleep_Spike_OccRate.png")
        #plt.waitforbuttonpress()
        plt.close()

        patient_occ_rates_df = pd.DataFrame(patient_occ_rates)
        pass

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
        all_stages_szr_auroc = prediction_results_df[prediction_results_df.Stage=='AllStages'].groupby('HasSeizures').Value.get_group(True).values

        n2_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N2'].groupby('HasSeizures').Value.get_group(False).values
        n3_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N3'].groupby('HasSeizures').Value.get_group(False).values
        n1_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='N1'].groupby('HasSeizures').Value.get_group(False).values
        rem_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='REM'].groupby('HasSeizures').Value.get_group(False).values
        wake_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='Wake'].groupby('HasSeizures').Value.get_group(False).values
        all_stages_noszr_auroc = prediction_results_df[prediction_results_df.Stage=='AllStages'].groupby('HasSeizures').Value.get_group(False).values

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
        mu_u, mu_p = stats.mannwhitneyu(all_stages_szr_auroc, all_stages_noszr_auroc, alternative='two-sided', use_continuity=True, method='auto')
        print(f"All Stages AUROC Mann-Whitney U test: mu = {mu_u:.2f}, p-value = {mu_p:.2e}")

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
        axs.set_title(f"{self.study_name}\nSOZ Prediction Performance vs. Nr. Clinical Seizures\nNr.Patients = {nr_pats}", fontsize=24)
        axs.set_ylabel("AUROC", fontsize=24)
        axs.set_xlabel("Nr. Clinical Seizures", fontsize=24)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_Prediction_Performance_vs_SZR_Count.png")
        #plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":

    # Predict SOZ based on Spike Activity
    studies_ls =  [fr_ILAES2025_patients(), ACH_Pediatric_Patients_All()]
    #studies_ls =  [ACH_Pediatric_Patients_All()]

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

        characterization_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Spike_Characterized_Channels")
        stages_spikes_duration_rate_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Stage_Spike_Occurrence_Rate")

        an_sleep_stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        stages_colors = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}
        for k,v in stages_colors.items():
            stages_colors[k] = (v[0]/255, v[1]/255, v[2]/255)

        spike_analyzer = Spike_Activity_Analyzer(study.dataset_name, characterization_datapath, stages_spikes_duration_rate_datapath, pats_ls, an_sleep_stages_ls, stages_colors, images_opath, szr_cnt_df)

        # Read and analyze sleep stages and spike occurrence rate data
        stage_duration_spike_rate_df = spike_analyzer.read_stages_duration_and_spike_rates()
        spike_analyzer.plot_group_sleep_stage_durations(stage_duration_spike_rate_df)
        spike_analyzer.plot_individual_sleep_stage_durations(stage_duration_spike_rate_df)
        spike_analyzer.plot_spike_occ_rate(stage_duration_spike_rate_df)
        spike_analyzer.analyze_spike_occ_rate_wake_sleep(stage_duration_spike_rate_df)
        
        # Read and analyze spike actvity data (average spike amplitude)
        spike_data_df = spike_analyzer.read_patient_spike_data()
        spike_analyzer.plot_per_stage_spike_activity(spike_data_df)
        spike_data_df = spike_analyzer.handle_patient_outliers(spike_data_df.copy())

        spike_data_df = spike_analyzer.get_patient_scaled_spike_data(spike_data_df)

        spike_analyzer.hypothesis_test_soz_vs_nonsoz(spike_data_df)
        prediction_results_df = spike_analyzer.predict_soz_with_spike_activity(spike_data_df, add_features=False)
        spike_analyzer.plot_soz_prediction_performance_vs_szr_count(prediction_results_df)
        prediction_results_ls.append(prediction_results_df)


    for i, study in enumerate(studies_ls):
        print(f"\n\nStudy: {study.dataset_name}")
        prediction_results_df = prediction_results_ls[i]      
        print('Median AUROC')
        print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median())
        # print('Std_AUROC')
        # print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).std())

        # print('Median MCC')
        # print(prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).median())
        # print('Std_MCC')
        # print(prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).std())

