import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

FIGSIZE = (16, 8)
plt.style.use('seaborn-v0_8-darkgrid')

characterization_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Spike_Characterized_Channels")
#characterization_datapath = Path("C:\\Users\\HFO\\Development\\pyeeg_toolbox\\Vectorized_WdwAn_Output\\Spike_Characterized_Channels_NoNoise")

STAGES_COLORS = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}
for k,v in STAGES_COLORS.items():
    STAGES_COLORS[k] = (v[0]/255, v[1]/255, v[2]/255)

pats_ls = [
    "pat_FR_1125_AvgSpikeWdwActivity.csv",
    "pat_FR_253_AvgSpikeWdwActivity.csv",
    "pat_FR_548_AvgSpikeWdwActivity.csv",
    "pat_FR_970_AvgSpikeWdwActivity.csv",
    "pat_FR_1073_AvgSpikeWdwActivity.csv",
    "pat_FR_1084_AvgSpikeWdwActivity.csv",
    "pat_FR_1096_AvgSpikeWdwActivity.csv",
]

##################################################
# Concatenate data from all patients, scale amplitude for each patient
spike_data_df = pd.DataFrame()
for pdata_fn in pats_ls:
    data_fpath = characterization_datapath / pdata_fn
    print(data_fpath)
    pdata_df = pd.read_csv(data_fpath)
    pdata_df.Amplitude = MinMaxScaler().fit_transform(pdata_df.Amplitude.values.reshape(-1, 1))
    pdata_df['Patient'] = pdata_fn.replace('_AvgSpikeWdwActivity.csv', '')
    spike_data_df = pd.concat([spike_data_df, pdata_df])

spike_data_df.loc[spike_data_df.SOZ==1, 'SOZ'] = 'SOZ'
spike_data_df.loc[spike_data_df.SOZ==0, 'SOZ'] = 'Non-SOZ'

sleep_stages_ls = ['N3', 'N2', 'N1', 'REM', 'Wake']
#for stage_name in sleep_stages_ls:
sns.violinplot(data=spike_data_df, x='Stage', y='Amplitude', hue='Stage', split=False, inner='quartile')
#sns.boxplot(data=spike_data_df, x='Stage', y='Amplitude', hue='Stage', split=False, inner='quartile')
plt.title(f"Spike Activity in different States of Awareness\nNr. Patients={len(pats_ls)}")
plt.ylabel("Spike Activity \n(Scaled per Patient)")
plt.grid(color='0.8', linestyle='-', linewidth=0.5)
#plt.waitforbuttonpress()
plt.close()


##################################################
# Compare SOZ vs Non-SOZ
fig, axs = plt.subplots(1, 6, figsize=FIGSIZE)

# All Stages
plt_ax = axs[0]
res = stats.mannwhitneyu(spike_data_df.Amplitude[spike_data_df.SOZ=='SOZ'], spike_data_df.Amplitude[spike_data_df.SOZ=='Non-SOZ'])
nr_soz = len(spike_data_df[spike_data_df.SOZ=='SOZ'])
nr_non_soz = len(spike_data_df[spike_data_df.SOZ=='Non-SOZ'])

#sns.violinplot(data=spike_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=axs[0])
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
    #sns.violinplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=plt_ax)
    sns.boxplot(data=stage_data_df, x='SOZ', y='Amplitude', hue='SOZ', ax=plt_ax)

    plt_ax.set_title(f"{stage_name}\n p-value = {res.pvalue:.2e}\n nr.SOZ={nr_soz}, nr.NonSOZ={nr_non_soz}")
    plt_ax.set_ylabel("Spike Activity \n(Scaled per Patient)")
    plt_ax.grid(color='0.8', linestyle='-', linewidth=0.5)


plt.get_current_fig_manager().full_screen_toggle()
plt.suptitle(f"Spike Activity in SOZ vs. Non-SOZ")
plt.tight_layout()
#plt.waitforbuttonpress()
plt.close()

pass

def oversample_patients(train_set_df):
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
##################################################
# Predict SOZ based on Spike Activity
prediction_results = {'Stage':[], 'Metric':[], 'Value':[]}
for si, stage_name in enumerate(sleep_stages_ls):
    stage_data_df = spike_data_df[spike_data_df.Stage==stage_name]
    for pat_id in stage_data_df.Patient.unique():
        train_set_df = stage_data_df[stage_data_df.Patient!=pat_id]
        test_set_df = stage_data_df[stage_data_df.Patient==pat_id]

        # scaler = StandardScaler()
        # train_set_df.Amplitude = scaler.fit_transform(train_set_df.Amplitude.values.reshape(-1, 1))
        # test_set_df.Amplitude = scaler.transform(test_set_df.Amplitude.values.reshape(-1, 1))

        train_set_df = oversample_patients(train_set_df)
        
        X_train = train_set_df.Amplitude.to_numpy().reshape(-1, 1)
        y_train = train_set_df.SOZ.to_numpy()=='SOZ'

        # Oversample to have equal positives and negatives

        
        X_test = test_set_df.Amplitude.to_numpy().reshape(-1, 1)
        y_test = test_set_df.SOZ.to_numpy()=='SOZ'

        # X_train = np.power(X_train,2).reshape(-1, 1)
        # X_test = np.power(X_test,2).reshape(-1, 1)

        # X_train = np.column_stack([X_train, X_train**2, X_train**3])
        # X_test = np.column_stack([X_test, X_test**2, X_test**3])


        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        print("Training Set", len(y_test))

        #model = SVC(random_state=42).fit(X_train, y_train)
        #y_predicted = model.predict(X_test) 

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

print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median())
print(prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).std())
pass

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='AUROC'], x='Stage', y='Value', hue='Stage', palette=STAGES_COLORS, ax=axs[0])
axs[0].set_title("Area under the ROC Curve")
axs[0].set_ylabel("AUROC")
medians_df = prediction_results_df[prediction_results_df.Metric=='AUROC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
vertical_offset = prediction_results_df[prediction_results_df.Metric=='AUROC'].Value.median() * 0.05 # offset from median for display
for xtick in box_plot.get_xticks():
    median_val = medians_df.Value[medians_df.Stage==sleep_stages_ls[xtick]].to_numpy()[0]
    median_str = f"{median_val:.2f}"
    box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size='large',color='w',weight='semibold')
    pass

box_plot = sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='MCC'], x='Stage', y='Value', hue='Stage', palette=STAGES_COLORS, ax=axs[1])
axs[1].set_title("Matthews Correlation Coefficient")
axs[1].set_ylabel("MCC")
medians_df = prediction_results_df[prediction_results_df.Metric=='MCC'][['Stage', 'Value']].groupby(['Stage']).median().reset_index()
vertical_offset = prediction_results_df[prediction_results_df.Metric=='MCC'].Value.median() * 0.05 # offset from median for display
for xtick in box_plot.get_xticks():
    median_val = medians_df.Value[medians_df.Stage==sleep_stages_ls[xtick]].to_numpy()[0]
    median_str = f"{median_val:.2f}"
    box_plot.text(x= xtick, y=median_val, s=median_str, horizontalalignment='center',size='x-large',color='w',weight='semibold')
    pass

plt.waitforbuttonpress()
plt.close()

##################################################
# Post-hoc analysis
# Compare SOZ prediction scores when usings spike activity from different sleep stages

metric_name = 'AUROC'
p_vals  ={'Metric':[], 'Groups':[], 'pvalue':[]}
for stage_a in sleep_stages_ls:
    for stage_b in sleep_stages_ls:
        if stage_a==stage_b:
            continue
        data_a = prediction_results_df[np.logical_and(prediction_results_df.Metric==metric_name, prediction_results_df.Stage==stage_a)].Value.to_numpy()
        data_b = prediction_results_df[np.logical_and(prediction_results_df.Metric==metric_name, prediction_results_df.Stage==stage_b)].Value.to_numpy()
        res = stats.wilcoxon(data_a, data_b, alternative='two-sided', method='exact')
        p_vals['Metric'].append(metric_name)
        p_vals['Groups'].append(f"{stage_a} vs. {stage_b}")
        p_vals['pvalue'].append(res.pvalue)

metric_name = 'MCC'
for stage_a in sleep_stages_ls:
    for stage_b in sleep_stages_ls:
        if stage_a==stage_b:
            continue

        data_a = prediction_results_df[np.logical_and(prediction_results_df.Metric==metric_name, prediction_results_df.Stage==stage_a)].Value.to_numpy()
        data_b = prediction_results_df[np.logical_and(prediction_results_df.Metric==metric_name, prediction_results_df.Stage==stage_b)].Value.to_numpy()
        res = stats.wilcoxon(data_a, data_b, alternative='two-sided', method='exact')
        p_vals['Metric'].append(metric_name)
        p_vals['Groups'].append(f"{stage_a} vs. {stage_b}")
        p_vals['pvalue'].append(res.pvalue)

p_vals_df = pd.DataFrame(p_vals)
print(p_vals_df)

print((p_vals_df.pvalue<0.01).sum())

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='AUROC'], x='Stage', y='Value', hue='Stage', palette=STAGES_COLORS, ax=axs[0])
axs[0].set_title("Area under the ROC Curve")
axs[0].set_ylabel("AUROC")

sns.boxplot(data=prediction_results_df[prediction_results_df.Metric=='MCC'], x='Stage', y='Value', hue='Stage', palette=STAGES_COLORS, ax=axs[1])
axs[1].set_title("Matthews Correlation Coefficient")
axs[1].set_ylabel("MCC")

plt.waitforbuttonpress()
plt.close()
pass