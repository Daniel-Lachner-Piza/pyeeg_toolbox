def get_files_in_folder(eeg_data_path:str=None, file_extension:str='.lay') -> None:
    """
    This function retrieves all files with a specific extension from a given directory.

    Parameters:
    file_extension (str): The file extension to filter for. Default is '.lay'.

    Returns:
    None
    """
    pat_files_ls = [fn for fn in eeg_data_path.glob(f"*{file_extension}")]
    # Check if any files were found
    assert len(pat_files_ls)>0, f"No {file_extension} files in folder {eeg_data_path}"
    return pat_files_ls