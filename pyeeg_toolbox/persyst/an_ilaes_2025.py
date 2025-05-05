import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from studies_info import fr_ILAES2025_patients, ACH_Pediatric_Patients

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")

FIGSIZE = (16, 8)

class Spike_Activity_Analyzer:
    """
    Class to analyze spike activity in EEG data, including reading data, handling outliers, scaling, and plotting results.
    """

    def __init__(self, characterization_datapath:str=None, stages_spikes_duration_rate_datapath:str=None, pats_ls:list[str]=None, sleep_stages_ls:list[str]=None, stages_colors:dict=None, images_output_path:str=None):
        self.characterization_datapath = characterization_datapath
        self.stages_spikes_duration_rate_datapath = stages_spikes_duration_rate_datapath
        self.pats_ls = pats_ls
        self.sleep_stages_ls = sleep_stages_ls
        self.stages_colors = stages_colors
        self.images_output_path = images_output_path
        

    def read_patient_spike_data(self):
        nr_pats = len(self.pats_ls)
        # Concatenate data from all patients, scale amplitude for each patient
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
            z_outliers_thresh = 3 # 3.0 * pdata_df.Amplitude.std()
            outliers_thresh_b = pdata_df.Amplitude.mean() + z_outliers_thresh * pdata_df.Amplitude.std()
            
            # Remove outliers from the data using percentile method
            outliers_thresh_c = np.percentile(pdata_df.Amplitude, 99.0)# 2.5 * pdata_df.Amplitude.std()

            outliers_thresh = outliers_thresh_c
            # outliers_thresh = np.min([outliers_thresh_a, outliers_thresh_b, outliers_thresh_c])
            # outliers_thresh = np.max([outliers_thresh_a, outliers_thresh_b, outliers_thresh_c])
            # outliers_thresh = np.mean([outliers_thresh_a, outliers_thresh_b, outliers_thresh_c])

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
            pdata_df.Amplitude = StandardScaler().fit_transform(pdata_df.Amplitude.values.reshape(-1, 1)) # MinMaxScaler, StandardScaler()
            scaled_spike_data_df = pd.concat([scaled_spike_data_df, pdata_df])

        return scaled_spike_data_df


    def plot_sleep_stage_durations(pats_ls:list[str]=None, images_output_path:str=None):
        nr_pats = len(pats_ls)
        spike_occ_rate_pats_ls = [pn + "_StageSpikeOccurrenceRate.csv" for pn in pats_ls]
        stage_spike_data = pd.DataFrame()
        for pdata_fn in spike_occ_rate_pats_ls:
            data_fpath = stages_spikes_duration_rate_datapath / pdata_fn
            #print(data_fpath)
            try:
                pdata_df = pd.read_csv(data_fpath)
                stage_spike_data = pd.concat([stage_spike_data, pdata_df])
            except:
                print(f"File {pdata_fn} not found")

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
            stage_sel = stage_spike_data.Stage==stage_name
            assert stage_sel.sum() == nr_pats, "More than one entry per patient"
            sum_stages_dur_mins.append(stage_spike_data.StageDurM[stage_sel].sum())
            pass

        to_plot_stages_colors = [self.stages_colors[k] for k in to_plot_stage_names]

        sum_stages_dur_perc = (np.array(sum_stages_dur_mins)/np.sum(sum_stages_dur_mins))*100
        wedgeprops = {"edgecolor" : "white", 'linewidth': 5, 'antialiased': True}
        
        patches, texts, pcts = axs.pie(x=sum_stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', textprops={'fontsize':24, 'color':"w", 'weight':'bold'}, startangle=-200)
        for i, patch in enumerate(patches):
            texts[i].set_color(patch.get_facecolor())
        axs.set_ylabel("Relative Duration of Sleep Stages (%)", fontsize=24)
        axs.set_title(f"Proportion of summed duration of Sleep Stages\nNr.Patients = {nr_pats}", fontsize=24, color='black')
        pass
        #plt.legend(loc='lower right', fontsize=24)

        # Overlay image on plot
        im_width, im_height = sleep_ref_img.size
        bbox = fig.get_window_extent() 
        fig.figimage(sleep_ref_img, xo=int(bbox.x1-im_width/2), yo=int(bbox.y1-im_height/2), zorder=3, alpha=.7, origin='upper')

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(images_output_path / "Duration_Of_Sleep_Stages.png")
        #plt.waitforbuttonpress()
        plt.close()


    def plot_spike_occ_rate(self, spike_data_df:pd.DataFrame=None):
        nr_pats = len(pats_ls)
        spike_occ_rate_pats_ls = [pn + "_StageSpikeOccurrenceRate.csv" for pn in pats_ls]
        stage_spike_data = pd.DataFrame()
        for pdata_fn in spike_occ_rate_pats_ls:
            data_fpath = stages_spikes_duration_rate_datapath / pdata_fn
            #print(data_fpath)
            try:
                pdata_df = pd.read_csv(data_fpath)
                stage_spike_data = pd.concat([stage_spike_data, pdata_df])
            except:
                print(f"File {pdata_fn} not found")

        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        assert nr_pats == len(stage_spike_data.PatID.unique()), "More than one entry per patient"
        sns.violinplot(data=stage_spike_data, x='Stage', y='SpikeOccRate', hue='Stage', palette=self.stages_colors, ax=axs)
        axs.set_ylabel("Spikes Activity (uV)", fontsize=24)
        axs.set_title(f"Average Spike Activity\nNr.Patients = {nr_pats}", fontsize=24)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig(images_output_path / "Average_Spike_Activity.png")
        #plt.waitforbuttonpress()
        plt.close()


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
        for si, stage_name in enumerate(sleep_stages_ls):
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
        plt.suptitle(f"Spike Activity in SOZ vs. Non-SOZ\nNr. Patients={nr_pats}")
        plt.tight_layout()
        plt.savefig(images_output_path / "Spike_Activity_SOZ_vs_NonSOZ_Hypothesis_Tests.png")
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

    def predict_soz_with_spike_occ_rate(self, spike_data_df:pd.DataFrame=None):
        
        nr_pats = len(spike_data_df.Patient.unique())

        # Predict SOZ based on Spike Activity
        prediction_results = {'Stage':[], 'Metric':[], 'Value':[]}
        for si, stage_name in enumerate(sleep_stages_ls):
            stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]
            for pat_id in stage_data_df.Patient.unique():
                train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
                test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

                # Oversample to have equal positives and negatives
                #train_set_df = self.oversample_patients(train_set_df)

                X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_train = train_set_df.SOZ.to_numpy()=='SOZ'
                #print(f"Positives to Negatives Ratio:{np.sum(y_train)/len(y_train)*100}")
                
                X_test = test_set_df.Amplitude.to_numpy().reshape(-1, 1)
                y_test = test_set_df.SOZ.to_numpy()=='SOZ'

                # if np.unique(y_test).shape[0] == 1:
                #     print(f"Patient {pat_id} has only one class in test set")
                #     continue

                assert np.unique(y_train).shape[0] > 1, "Train set has only one class"
                assert np.unique(y_test).shape[0] > 1, "Test set has only one class"

                # Basic feature engineering
                X_train = np.hstack([X_train, X_train**2, X_train**3])
                X_test = np.hstack([X_test, X_test**2, X_test**3])

                model = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', max_iter=10000, tol=0.01)

                model.fit(X_train, y_train)
                y_predicted = model.predict(X_test)

                # model = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', n_jobs=-1)
                # model.fit(X_train, y_train)
                # y_predicted = model.predict(X_test)

                # model = SVC()
                # model.fit(X_train, y_train)
                # y_predicted = model.predict(X_test)

                print("Training Set", len(y_test))

                auroc_val = roc_auc_score(y_test, y_predicted)
                mcc_val = matthews_corrcoef(y_test, y_predicted)
                prediction_results['Stage'].append(stage_name)
                prediction_results['Metric'].append('AUROC')
                prediction_results['Value'].append(auroc_val)
                prediction_results['Stage'].append(stage_name)
                prediction_results['Metric'].append('MCC')
                prediction_results['Value'].append(mcc_val)

        prediction_results_df = pd.DataFrame(prediction_results)

        for sleep_stage in sleep_stages_ls:
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

        fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
        box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='AUROC'], x='Stage', y='Value', hue='Stage', palette=self.stages_colors, ax=axs[0])
        axs[0].set_title(f"Area under the ROC Curve\nNr. Patients={nr_pats}")
        axs[0].set_ylabel("AUROC")
        medians_df = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
        vertical_offset = prediction_results_df[prediction_results_df.Metric=='AUROC'].Value.median() * 0.05 # offset from median for display
        for xtick in box_plot.get_xticks():
            median_val = medians_df.Value[medians_df.Stage==sleep_stages_ls[xtick]].to_numpy()[0]
            median_str = f"{median_val:.2f}"
            box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size='large',color='w',weight='semibold')
            pass

        box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='MCC'], x='Stage', y='Value', hue='Stage', palette=self.stages_colors, ax=axs[1])
        axs[1].set_title(f"Matthews Correlation Coefficient\nNr. Patients={nr_pats}")
        axs[1].set_ylabel("MCC")
        medians_df = prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
        vertical_offset = prediction_results_df[prediction_results_df.Metric=='MCC'].Value.median() * 0.05 # offset from median for display
        for xtick in box_plot.get_xticks():
            median_val = medians_df.Value[medians_df.Stage==sleep_stages_ls[xtick]].to_numpy()[0]
            median_str = f"{median_val:.2f}"
            box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size='x-large',color='w',weight='semibold')
            pass

        plt.get_current_fig_manager().full_screen_toggle()
        plt.suptitle(f"Prediction of SOZ\nNr. Patients={nr_pats}")
        plt.tight_layout()
        plt.savefig(images_output_path / "SOZ_Prediction.png")
        #plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":

    study = ACH_Pediatric_Patients() # ACH_Pediatric_Patients, fr_ILAES2025_patients
    if 'freiburg' in str(study.eeg_data_path).lower():
        images_output_path = Path(os.getcwd()) / "Images_Output"
    else:
        images_output_path = Path(os.getcwd()) / "Images_Output_Pediatric"
    os.makedirs(images_output_path, exist_ok=True)

    pats_ls = list(study.patients.keys())

    characterization_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Spike_Characterized_Channels")
    stages_spikes_duration_rate_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Stage_Spike_Occurrence_Rate")

    sleep_stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
    stages_colors = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}
    for k,v in stages_colors.items():
        stages_colors[k] = (v[0]/255, v[1]/255, v[2]/255)

    spike_analyzer = Spike_Activity_Analyzer(characterization_datapath, stages_spikes_duration_rate_datapath, pats_ls, sleep_stages_ls, stages_colors, images_output_path)

    spike_data_df = spike_analyzer.read_patient_spike_data()
    spike_data_df = spike_analyzer.handle_patient_outliers(spike_data_df)
    spike_data_df = spike_analyzer.get_patient_scaled_spike_data(spike_data_df)

    spike_analyzer.hypothesis_test_soz_vs_nonsoz(spike_data_df)
    spike_analyzer.predict_soz_with_spike_occ_rate(spike_data_df)

    #spike_analyzer.plot_spike_occ_rate(spike_data_df)
    #plot_sleep_stage_durations(pats_ls, images_output_path)
