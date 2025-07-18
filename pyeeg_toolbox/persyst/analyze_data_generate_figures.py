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
        for pdata_fn in pats_ls:
            pdata_df = spike_data_df[spike_data_df.Patient.str.fullmatch(pdata_fn, case=False)].reset_index(drop=True).copy()
            pdata_df.Amplitude = MinMaxScaler().fit_transform(pdata_df.Amplitude.values.reshape(-1, 1)) # MinMaxScaler, StandardScaler()
            scaled_spike_data_df = pd.concat([scaled_spike_data_df, pdata_df])

        return scaled_spike_data_df


    def hypothesis_test_soz_vs_nonsoz(self, spike_data_df:pd.DataFrame=None):

        nr_pats = len(spike_data_df.Patient.unique())

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, 5, figsize=FIGSIZE)


        res_dict = {'Stage':[], 'U_statistic':[], 'p_value':[]}

        # Compare SOZ vs Non-SOZ in the different sleep stages
        for si, stage_name in enumerate(self.sleep_stages_ls):
            plt_ax = axs[si]
            stage_data_df = spike_data_df[spike_data_df.Stage==stage_name].copy().reset_index(drop=True)
            stage_data_df.loc[:, 'Amplitude'] = stage_data_df.Amplitude.values*1000*1000 # convert to uV
            res = stats.mannwhitneyu(stage_data_df.Amplitude[stage_data_df.SOZ=='SOZ'], stage_data_df.Amplitude[stage_data_df.SOZ=='Non-SOZ'])
            nr_soz = len(stage_data_df[stage_data_df.SOZ=='SOZ'])
            nr_non_soz = len(stage_data_df[stage_data_df.SOZ=='Non-SOZ'])
            res_dict['Stage'].append(stage_name)
            res_dict['U_statistic'].append(res.statistic)
            res_dict['p_value'].append(res.pvalue)
            #sns.boxplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=plt_ax)
            
            colors_soz = [c for c in self.stages_colors[stage_name]]
            colors_soz.append(1)  # Add alpha channel for transparency
            colors_non_soz = [c for c in self.stages_colors[stage_name]]
            colors_non_soz.append(0.5)  # Add alpha channel for transparency
            soz_colors_dict = {'SOZ':colors_soz, 'Non-SOZ':colors_non_soz}
            sns.boxplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', palette=soz_colors_dict, showfliers=False, ax=plt_ax)
            medians_df = stage_data_df[['SOZ', 'Amplitude']].groupby(['SOZ']).median().reset_index()

            for xti, xtick in enumerate(plt_ax.get_xticks()):
                median_val = medians_df.Amplitude.values[xti]
                median_str = f"{median_val:.0f}"
                plt_ax.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size=32, color='k', weight='semibold') # size=10
                pass

            for pi, patch in enumerate(plt_ax.patches):
                if pi == 0:
                    # r, g, b, a = patch.get_facecolor() # Get current color (including alpha)
                    #patch.set_facecolor((r, g, b, 0.5))
                    patch.set_hatch('..')  # Add hatch pattern to differentiate SOZ and Non-SOZ
                    fc = patch.get_facecolor()
                    patch.set_edgecolor(fc)
                    patch.set_facecolor('none')

            if si == 0:
                plt_ax.set_ylabel("Spike Activity (uV)", fontsize=32)
            else:
                plt_ax.set_ylabel("")
                plt_ax.set_yticklabels("")

            plt_ax.set_xlabel("")
            plt_ax.tick_params(axis='x', labelsize=32, rotation=60)
            plt_ax.tick_params(axis='y', labelsize=32)

            if 'Freiburg' in self.study_name:
                plt_ax.set_ylim(0, 90)
            else:
                plt_ax.set_ylim(0, 310)
            plt_ax.set_title(f"{stage_name}", fontsize=32, color=self.stages_colors[stage_name], weight='bold')
            # plt.xticks(fontsize=32)
            # plt.yticks(fontsize=32)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}", fontsize=48)
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Spike_Activity_SOZ_vs_NonSOZ_Hypothesis_Tests.png")
        #plt.waitforbuttonpress()
        plt.close()
        #print(res_dict) 
        pass

    def predict_soz_with_spike_activity(self, spike_data_df:pd.DataFrame=None, add_features:bool=False):
        
        nr_pats = len(spike_data_df.Patient.unique())

        # Predict SOZ based on Spike Activity
        analysis_stages = copy.copy(self.sleep_stages_ls)
        analysis_stages_colors = copy.copy(self.stages_colors)
        # analysis_stages.insert(0, 'AllStages')
        # analysis_stages_colors['AllStages'] = (0.5, 0.5, 0.5)
        prediction_results = {'TestPatID':[], 'TestPatient':[], 'Stage':[], 'Metric':[], 'Value':[], 'NrClinicalSzrs':[], 'NrElectroSzrs':[]}

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, len(analysis_stages), figsize=FIGSIZE, constrained_layout=True)

        auroc_dict = {stage:[] for stage in analysis_stages}
        for si, stage_name in enumerate(analysis_stages):
            if stage_name == 'AllStages':
                stage_data_df = spike_data_df[spike_data_df.Stage!='Unknown']
            else:
                stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]

            pass

            roc_avg = {'fpr':np.linspace(0,1,1000), 'tpr':np.zeros_like(np.linspace(0,1,1000))}
            auroc_ls = []
            for pidx, pat_id in enumerate(stage_data_df.Patient.unique()):
                train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
                test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

                X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_train = train_set_df.SOZ.to_numpy()=='SOZ'
                X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
                #print(f"Positives to Negatives Ratio:{np.sum(y_train)/len(y_train)*100}")
                
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

                # Get probabilities for the positive class
                y_predicted_proba = model.predict_proba(X_test)[:, 1]
                # Calculate ROC Curve points
                fpr, tpr, thresholds = roc_curve(y_test, y_predicted_proba)
                # Calculate AUC
                y_predicted_bin = model.decision_function(X_test)

                auc_score = roc_auc_score(y_test, y_predicted_proba)  # Use a threshold of 0.5 for binary classification
                #auroc_val = roc_auc_score(y_test, y_predicted_bin)
                auroc_ls.append(auc_score)
                auroc_dict[stage_name].append(auc_score)
                # print(f"Binary AUC Score: {auroc_val:.2f}")
                # print(f"Probab. AUC Score: {auc_score:.2f}")

                # interpolate tpr to have 100 points
                tpr_interp = np.interp(roc_avg['fpr'], fpr, tpr)

                roc_avg['tpr'] = np.add(roc_avg['tpr'], tpr_interp)
                
                pat_szrcnt_df = self.szr_cnt_df.loc[self.szr_cnt_df.PatID.str.fullmatch(pat_id, case=False)]

                label=f"{pat_id}(AUC:{auc_score}:.2f)"
                axs[si].plot(roc_avg['fpr'], tpr_interp, color=analysis_stages_colors[stage_name], alpha=0.5, linewidth=1, linestyle='-')                
                pass

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

            label=f"Avg.ROC(Avg.AUC: {np.mean(auroc_ls):.2f})"
            axs[si].plot(roc_avg['fpr'], roc_avg['tpr'],color=analysis_stages_colors[stage_name], alpha=1, linewidth=8, linestyle='-')
            axs[si].plot(np.linspace(0,1,100), np.linspace(0,1,100),color='k', alpha=1, linewidth=1, linestyle='--')

            axs[si].set_title(f"{stage_name}", fontsize=48)
            
            if si==0:
                axs[si].set_ylabel("TPR", fontsize=32)
            else:
                axs[si].set_ylabel("")
                axs[si].set_yticklabels("")
            axs[si].set_xlabel("FPR", fontsize=32)

            axs[si].set_xticks(np.linspace(0, 1, 5))
            axs[si].set_yticks(np.linspace(0, 1, 5))
            axs[si].tick_params(axis='x', rotation=45, labelsize=32)          
            axs[si].tick_params(axis='y', rotation=0, labelsize=32)

            # Add text with a text box to the subplot
            text_str = f"AUROC\nAvg.: {np.mean(auroc_ls):.2f}\nStd.Dev.: {np.std(auroc_ls):.2f}"
            axs[si].text(0.5, 0.98, text_str,
                    transform=axs[si].transAxes, # Use axes coordinates
                    fontsize=32,
                    verticalalignment='top',
                    horizontalalignment='center',
                    bbox={'facecolor': analysis_stages_colors[stage_name], 'alpha': 0.7, 'pad': 5})
            print(f"Stage: {stage_name}, AUROC: {np.mean(auroc_ls):.3f} (std.dev.: {np.std(auroc_ls):.3f})")       
                
        plt.get_current_fig_manager().full_screen_toggle()
        plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9, bottom=0.3, top=0.7)
        #plt.show()
        plt.savefig(self.images_output_path / "SOZ_Prediction_ROC_Curves.png")
        plt.close()
        pass


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

        # Conduct the repeated measures ANOVA 
        test_prediction_results_df = prediction_results_df[prediction_results_df.Metric=='AUROC'].reset_index(drop=True).copy()
        anova_results = AnovaRM(data=test_prediction_results_df, subject='TestPatient', depvar='Value', within=['Stage']).fit()
        print(anova_results)
        p_val_rmanova = anova_results.anova_table['Pr > F'].mean()

        #all_stages_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('AllStages').values
        n3_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N3').values
        n2_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N2').values
        n1_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('N1').values
        rem_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('REM').values
        wake_auroc = test_prediction_results_df.groupby('Stage').Value.get_group('Wake').values

        stat_val, p_val = stats.kruskal(n3_auroc, n2_auroc, n1_auroc, rem_auroc, wake_auroc)
        stat_val, p_val = stats.kruskal(auroc_dict['N3'], auroc_dict['N2'], auroc_dict['N1'], auroc_dict['REM'], auroc_dict['Wake'])
        pass

        ###########################
        test_results = np.ones((len(analysis_stages),len(analysis_stages)))
        for ia, stage_name_a in enumerate(analysis_stages):
            aurocs_a = auroc_dict[stage_name_a]
            for ib, stage_name_b in enumerate(analysis_stages):
                aurocs_b = auroc_dict[stage_name_b]
                if ia!= ib:
                    # wilcoxon_stat, wilcoxon_p_val = stats.wilcoxon(aurocs_a, aurocs_b, nan_policy='raise', alternative='two-sided')
                    # test_results[ia,ib] = wilcoxon_p_val
                    t_stat, wilcoxon_p_val = stats.ttest_rel(aurocs_a, aurocs_b, nan_policy='raise', alternative='two-sided')
                    test_results[ia,ib] = wilcoxon_p_val

        # Create a mask
        mask = np.triu(np.ones_like(test_results, dtype=bool))
        threshold = 0.05 / ((len(analysis_stages) * (len(analysis_stages) - 1))/2)  # Bonferroni correction for multiple comparisons
        threshold = 0.05 / (len(analysis_stages)-1)  # Bonferroni correction for multiple comparisons
        print(f"Bonferroni corrected threshold: {threshold:.3f}")
                             
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        ax = sns.heatmap(test_results, vmin=0, vmax=threshold, center=threshold, mask=mask, cmap='coolwarm', annot=True, fmt=".3f", annot_kws={"size": 32}, linewidths=.5, linecolor='white', cbar_kws={"shrink": .8},ax=axs)
        cbar = ax.collections[0].colorbar
        # Adjust the font size of the colorbar tick labels
        cbar.ax.tick_params(labelsize=32) # Set specific font size
        cbar.set_label('p value', fontsize=32) # Set colorbar label

        #ax = sns.heatmap(test_results, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
        ax.grid(False)
        ax.set_xticklabels(analysis_stages, rotation=45, fontsize=32)
        ax.set_yticklabels(analysis_stages, rotation=0, fontsize=32)
        alpha_str = r" $\alpha$"        
        plt.title(f"Spike Activity\nWilcoxon Signed-Rank Test p-values\n({alpha_str}:{threshold})", fontsize=36)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_Localization_Wilcoxon_Test_Results.png")
        plt.close()


    ############################

        fig, axs = plt.subplots(1, 1, figsize=(4,8), sharey=True)
        #ax = axs[0]
        ax = axs
        box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='AUROC'], x='Stage', y='Value', hue='Stage', palette=analysis_stages_colors, ax=ax)
        axs.set_title(f"Area under the ROC Curve\nNr. Patients={nr_pats}", fontsize=48)
        axs.set_ylabel("AUROC", fontsize=32)
        axs.set_xlabel("")
        ax.set_xticklabels(analysis_stages, fontsize=32)
        #axs.tick_params(axis='x', rotation=0, labelsize=32)
        axs.tick_params(axis='y', rotation=0, labelsize=32)

        medians_df = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
        vertical_offset = prediction_results_df[prediction_results_df.Metric=='AUROC'].Value.median() * 0.05 # offset from median for display
        for xtick in box_plot.get_xticks():
            median_val = medians_df.Value[medians_df.Stage==analysis_stages[xtick]].to_numpy()[0]
            median_str = f"{median_val:.2f}"
            box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size=32,color='w',weight='semibold') # size=10
            pass

        #box_plot.text(x=box_plot.get_xticks()[-1], y=0.95, s=f"Repeated Measures Anova p_val: {p_val_rmanova:.2f}", horizontalalignment='center',size=14,color='r',weight='bold')


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

        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"{self.study_name}\nPrediction of SOZ", fontsize=48)
        plt.tight_layout()
        plt.savefig(self.images_output_path / "SOZ_Prediction.png")
        #plt.waitforbuttonpress()
        plt.close()

        return prediction_results_df

    def predict_soz_with_spike_activity_figure_1(self, spike_data_df:pd.DataFrame=None, add_features:bool=False):
        
        nr_pats = len(spike_data_df.Patient.unique())

        # Predict SOZ based on Spike Activity
        analysis_stages = copy.copy(self.sleep_stages_ls)
        analysis_stages_colors = copy.copy(self.stages_colors)
        prediction_results = {'TestPatID':[], 'TestPatient':[], 'Stage':[], 'Metric':[], 'Value':[], 'NrClinicalSzrs':[], 'NrElectroSzrs':[]}

        # Compare SOZ vs Non-SOZ
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)

        for si, stage_name in enumerate(analysis_stages):
            if stage_name == 'AllStages':
                stage_data_df = spike_data_df[spike_data_df.Stage!='Unknown']
            else:
                stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]
            pass

            roc_avg = {'fpr':np.linspace(0,1,1000), 'tpr':np.zeros_like(np.linspace(0,1,1000))}
            auroc_ls = []
            for pidx, pat_id in enumerate(stage_data_df.Patient.unique()):
                train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
                test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

                X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_train = train_set_df.SOZ.to_numpy()=='SOZ'
                X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
                #print(f"Positives to Negatives Ratio:{np.sum(y_train)/len(y_train)*100}")
                
                X_test = test_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_test = test_set_df.SOZ.to_numpy()=='SOZ'

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

            label=f"{stage_name}(avg. AUROC: {np.mean(auroc_ls):.2f})"
            axs.plot(roc_avg['fpr'], roc_avg['tpr'],color=analysis_stages_colors[stage_name], alpha=1, linewidth=8, linestyle='-',label=label)
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

            # # Add text with a text box to the subplot
            # text_str = f"SOZ Prediction\nROC  Curves}"
            # axs.text(0.5, 0.98, text_str,
            #         transform=axs.transAxes, # Use axes coordinates
            #         fontsize=32,
            #         verticalalignment='top',
            #         horizontalalignment='center',
            #         bbox={'facecolor': analysis_stages_colors[stage_name], 'alpha': 0.7, 'pad': 5})
            print(f"Stage: {stage_name}, AUROC: {np.mean(auroc_ls):.3f} (std.dev.: {np.std(auroc_ls):.3f})")       
        
        axs.set_title(f"SOZ Prediction\nROC Curves, All Sleep Stages", fontsize=48)
        axs.plot(np.linspace(0,1,100), np.linspace(0,1,100),color='k', alpha=1, linewidth=1, linestyle='--')    
        axs.legend(loc='lower right', fontsize=32, frameon=True, facecolor='w', edgecolor='k')
        axs.grid(True, linestyle='-', alpha=1, linewidth=1)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9, bottom=0.3, top=0.7)
        plt.savefig(self.images_output_path / "SOZ_Prediction_ROC_Curves_ver2.png")
        #plt.show()
        plt.close()
        pass
       
    

    def plot_group_sleep_stage_durations_piechart(self, stage_duration_spike_rate_df:pd.DataFrame=None):
        nr_pats = len(stage_duration_spike_rate_df.PatID.unique())

        sleep_ref_img_path = Path.cwd()/'pyeeg_toolbox/persyst/SleepStages_Reference.png'
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
        
        patches, texts, pcts = axs.pie(x=sum_stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':32, 'color':"w", 'weight':'bold'}, startangle=-200)
        for i, patch in enumerate(patches):
            texts[i].set_color(patch.get_facecolor())
        axs.set_ylabel("Relative Duration of Sleep Stages (%)", fontsize=32)
        axs.set_title(f"{self.study_name}\nProportion of summed duration of Sleep Stages\nNr.Patients = {nr_pats}", fontsize=48, color='black')
        pass
        #plt.legend(loc='lower right', fontsize=32)

        # Overlay image on plot
        im_width, im_height = sleep_ref_img.size
        bbox = fig.get_window_extent() 
        fig.figimage(sleep_ref_img, xo=int(bbox.x1-im_width/2), yo=int(bbox.y1-im_height/2), zorder=3, alpha=.7, origin='upper')

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(self.images_output_path / "Duration_Of_Sleep_Stages.png")
        #plt.waitforbuttonpress()
        plt.close()

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
        for stage_name in stage_duration_spike_rate_orig_df.Stage.unique():
            stage_sel = stage_duration_spike_rate_orig_df.Stage==stage_name
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
        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM']
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
        axs.set_ylabel("Spikes / electrode / min.", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        axs.tick_params(axis='x', labelsize=32)
        axs.tick_params(axis='y', labelsize=32)
        axs.set_ylim(0, 16) # set y-axis limit to 60 minutes

        # Plot Sleep Stages N2, N1, REM, Wake
        axs = all_axs[1]
        to_plot_stage_names = ['N2', 'N1', 'REM', 'Wake']
        stage_duration_spike_rate_df = stage_duration_spike_rate_orig_df.copy()
        stage_duration_spike_rate_df['StageDurH'] = stage_duration_spike_rate_df.StageDurM / 60.0 # convert to hours
        to_plot_stage_sel = stage_duration_spike_rate_df.Stage.isin(to_plot_stage_names)
        stage_duration_spike_rate_df = stage_duration_spike_rate_df[to_plot_stage_sel].reset_index(drop=True).copy()
        to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]
        assert nr_pats == len(stage_duration_spike_rate_df.PatID.unique()), "More than one entry per patient"
        bp_ax = sns.barplot(data=stage_duration_spike_rate_df, x='Stage', y='StageDurH', hue='Stage', 
            order=to_plot_stage_names, palette=to_plot_stages_colors, ax=axs,
            capsize=.2,
            errorbar=errorbar_def,
            err_kws=errorbar_characteristics,
            linewidth=1, edgecolor=".5", width=0.5, gap=0.1,
            estimator=np.mean
            )
        for cont in bp_ax.containers:
            axs.bar_label(cont, fmt='%.2f', fontsize=32, label_type='edge', padding=3, color='black', weight='bold')
        axs.set_ylabel("Spikes / electrode / min.", fontsize=32)
        axs.set_xlabel("Sleep Stage", fontsize=32)
        axs.tick_params(axis='x', labelsize=32)
        axs.tick_params(axis='y', labelsize=32)
        axs.set_ylim(0, 41) # set y-axis limit to 60 minutes

        plt.suptitle(f"{self.study_name}", fontsize=48)

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
            stage_sel = stage_duration_spike_rate_df.Stage==stage_name
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

    def plot_per_stage_spike_activity(self, spike_data_df:pd.DataFrame=None):

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
        for stage_name in all_pats_avg_stage_activity.Stage.unique():
            stage_sel = all_pats_avg_stage_activity.Stage==stage_name
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
        pass

    def analyze_sor_stages_differences(self, stage_duration_spike_rate_df:pd.DataFrame=None):
        
        # Analyze Differences in Spike Occurrence Rate between Sleep Stages
        patients_ls = list(stage_duration_spike_rate_df.PatID.unique())
        nr_pats = len(patients_ls)
        print(f"Spike Occurrence Rate Differences between Sleep Stages")
        print(f"Nr. Patients: {nr_pats}")

        stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        test_results = np.ones((len(stages_ls),len(stages_ls)))+100
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
                        wilcoxon_stat, p_val = stats.wilcoxon(spike_rate_a, spike_rate_b, nan_policy='raise', alternative='two-sided')
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

    def analyze_spike_activity_stages_differences(self, spike_data_df:pd.DataFrame=None):
        patients_ls = list(spike_data_df.Patient.unique())
        nr_pats = len(patients_ls)
        print(f"Analyzing Spike Activity differences between Sleep Stages")
        print(f"Nr. Patients: {nr_pats}")

        # Get average spike activity per stage for each patient
        all_pats_avg_stage_activity = spike_data_df[['Patient', 'Stage', 'Amplitude']].groupby(['Patient','Stage']).mean().reset_index()

        stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        test_results = np.ones((len(stages_ls),len(stages_ls)))+100
        for ia, stage_name_a in enumerate(stages_ls):
            stage_sel_a = all_pats_avg_stage_activity.Stage==stage_name_a
            spike_activity_a = all_pats_avg_stage_activity.Amplitude[stage_sel_a].to_numpy()
            assert stage_sel_a.sum() == nr_pats, "More than one entry per patient"
            for ib, stage_name_b in enumerate(stages_ls):
                stage_sel_b = all_pats_avg_stage_activity.Stage==stage_name_b
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
                        wilcoxon_stat, p_val = stats.wilcoxon(spike_activity_a, spike_activity_b, nan_policy='raise', alternative='two-sided')
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

    pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Accelerated")
    #pyeeg_output_path = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Output_HandleOffset_CorrectPolarity_NoAbsValue_Slow")

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
        characterization_datapath = pyeeg_output_path / "Spike_Characterized_Channels"
        stages_spikes_duration_rate_datapath = pyeeg_output_path / "Stage_Spike_Occurrence_Rate"
        an_sleep_stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
        stages_colors = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}
        for k,v in stages_colors.items():
            stages_colors[k] = (v[0]/255, v[1]/255, v[2]/255)
        spike_analyzer = Spike_Activity_Analyzer(study.dataset_name, characterization_datapath, stages_spikes_duration_rate_datapath, pats_ls, an_sleep_stages_ls, stages_colors, images_opath, szr_cnt_df)

        # Read and analyze sleep stages and spike occurrence rate data
        stage_duration_spike_rate_df = spike_analyzer.read_stages_duration_and_spike_rates()
        # Read and analyze spike actvity data (average spike amplitude)
        spike_data_df = spike_analyzer.read_patient_spike_data()

        # Analyze differences in sleep stage durations
        spike_analyzer.plot_group_sleep_stage_durations_barchart(stage_duration_spike_rate_df)
        #spike_analyzer.plot_individual_sleep_stage_durations(stage_duration_spike_rate_df)
                
        # Analyze differences in spike occurrence rate between sleep stages
        spike_analyzer.plot_spike_occ_rate(stage_duration_spike_rate_df)
        spike_analyzer.analyze_sor_stages_differences(stage_duration_spike_rate_df)

        # Analyze differences in spike activity between wake and sleep stages    
        spike_analyzer.plot_per_stage_spike_activity(spike_data_df)
        spike_analyzer.analyze_spike_activity_stages_differences(spike_data_df)
        
        # spike_data_df = spike_analyzer.handle_patient_outliers(spike_data_df.copy())
        # spike_analyzer.hypothesis_test_soz_vs_nonsoz(spike_data_df)

        # spike_data_df = spike_analyzer.get_patient_scaled_spike_data(spike_data_df)
        # prediction_results_df = spike_analyzer.predict_soz_with_spike_activity(spike_data_df, add_features=False)
        # spike_analyzer.predict_soz_with_spike_activity_figure_1(spike_data_df, add_features=False)
        # spike_analyzer.plot_soz_prediction_performance_vs_szr_count(prediction_results_df)
        # prediction_results_ls.append(prediction_results_df)

    sys.exit()

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

        print(f"\n\nAUROC Results")
        nr_pats = len(list(study.patients.keys()))
        stages_ci_ranges = {}
        for stage_name in prediction_results_df.Stage.unique():
            stage_sel = np.logical_and(prediction_results_df.Stage==stage_name, prediction_results_df.Metric=='AUROC')
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

