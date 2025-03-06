import pandas as pd


def correct_relabelled_chnames(chann_ls, pat_nr):
    ch_map_fpath = f"F:/FREIBURG_Simultaneous_OneHrFiles/iEEG_Seizure_Info/FR{pat_nr}_Chann_Map.csv"
    ch_map_df = pd.read_csv(ch_map_fpath)
    new_chann_ls = []
    for ch in chann_ls:
        mapped_chname = ch_map_df.NewChanName[ch_map_df.OrigChanName.str.fullmatch(ch, case=False)].values
        assert len(mapped_chname) == 1, f"Channel {ch} with several mappings or not found in the mapping file"
        new_chann_ls.append(mapped_chname[0])
    return new_chann_ls