import socket
import platform
from pathlib import Path

class EEG_Study_Info:
    def __init__(self) -> None:
        self.eeg_data_path = None
        self.sleep_data_path = None
        self.ispikes_data_path = None
        self.patients = None
        self.dataset_name = None

def get_system_info():
    sys_info={}
    sys_info['hostname']=socket.gethostname()
    sys_info['machine']=platform.machine()
    sys_info['system']=platform.system()
    return sys_info
        
def fr_four_patients():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "Freiburg_Four"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/FREIBURG_Simultaneous_OneHrFiles/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'pat_FR_253':['HRC5', 'HP1', 'HP2', 'HP3'],
        'pat_FR_970':['GC1'], 
        'pat_FR_1084':['M1', 'M2'], 
        'pat_FR_1096':['LDH1'],
        }
    
    return study_info
    
def fr_ILAES2025_patients():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "Freiburg_Epilepsiae_Simultaneous_17"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/FREIBURG_Simultaneous_OneHrFiles/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        #'pat_FR_139':[''], # N1 duration is 0 within 48 hours
        'pat_FR_253':['HRC5', 'HP1', 'HP2', 'HP3'],
        'pat_FR_264':[''], # SOZ channels are renamed
        'pat_FR_273':[''], # SOZ channels are renamed
        'pat_FR_384':[''],
        'pat_FR_375':[''],
        'pat_FR_442':[''],
        'pat_FR_548':[''],
        'pat_FR_565':[''],
        'pat_FR_583':[''],
        'pat_FR_590':[''],
        'pat_FR_862':[''],
        'pat_FR_916':[''],
        # 'pat_FR_922':[''], #duration of 24 hours only
        #'pat_FR_958':[''], # N1 duration is 0 within 48 hours
        'pat_FR_970':['GC1'], 
        'pat_FR_1073':[''],
        'pat_FR_1084':['M1', 'M2'], 
        'pat_FR_1096':['LDH1'],
        'pat_FR_1125':[''],
        }
    
    return study_info

def ACH_ILAES2025_patients():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "ACH_ILAES2025_patients"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/FREIBURG_Simultaneous_OneHrFiles/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'pat_FR_253':[],
        'pat_FR_264':[''], # Check sleep staging 
        'pat_FR_273':[''], # Check sleep staging
        'pat_FR_384':[''], # Check sleep staging
        'pat_FR_375':[''], # Check sleep staging
        'pat_FR_442':[''],
        'pat_FR_548':[''],
        'pat_FR_565':[''],
        'pat_FR_583':[''], # Check sleep staging
        'pat_FR_590':[''],
        'pat_FR_862':[''],
        'pat_FR_916':[''],
        'pat_FR_970':['GC1'], 
        'pat_FR_1073':[''],
        'pat_FR_1084':['M1', 'M2'], 
        'pat_FR_1096':['LDH1'],
        'pat_FR_1125':[''],
        #'pat_FR_139':[''], # N1 duration is 0 within 48 hours
        # 'pat_FR_922':[''], #duration of 24 hours only
        #'pat_FR_958':[''], # N1 duration is 0 within 48 hours
        }
    
    return study_info

def ACH_Pediatric_Patients():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "ACH_Pediatric_Patients"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/Pediatric_Patients_Simultaneous/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/Pediatric_Patients_Simultaneous/")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/Pediatric_Patients_Simultaneous/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'PAT001':[''],
        'PAT004':[''],
        'PAT005':[''], # check sleep staging
        'PAT006':[''],
        'PAT007':[''],
        }
    
    return study_info

def ACH_Pediatric_Patients_Spike_Drive():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "ACH_Pediatric_Patients"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("I:/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("E:/SimultEEG_PediatricPatients")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/I/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'PAT003':[''],
        'PAT009':[''],
        'PAT010':[''],
        'PAT011':[''],
        'PAT012':[''],
        }
    
    return study_info

def ACH_Pediatric_Patients_All():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "ACH_Pediatric_Patients"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/Pediatric_Patients_Simultaneous/")
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/Pediatric_Patients_Simultaneous/")
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/Pediatric_Patients_Simultaneous/")
    
    study_info.sleep_data_path = study_info.eeg_data_path
    study_info.ispikes_data_path = study_info.eeg_data_path
    study_info.channel_coordinates_data_path = study_info.eeg_data_path / "iEEG_Electrode_Coordinates"
    study_info.seizure_info_data_path = study_info.eeg_data_path / "iEEG_Seizure_Info"

    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'PAT001':[''],
        'PAT003':[''],
        'PAT004':[''],
        'PAT005':[''], # check sleep staging
        'PAT006':[''],
        'PAT007':[''],
        'PAT009':[''], # check sleep staging
        'PAT010':[''], # check sleep staging
        'PAT011':[''],
        'PAT012':[''],
        }
    
    return study_info