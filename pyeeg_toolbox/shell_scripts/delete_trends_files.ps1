$fr_pats_list = @('pat_FR_1084','pat_FR_1096','pat_FR_253','pat_FR_970')
$fr_pats_list = @('pat_FR_253')
$FreiburgFilesDirectory = 'F:\FREIBURG_Simultaneous_OneHrFiles\'
$to_delete_extensions = @( '.xml', '.mg2', '.af1', '.af2', '.indx', '.mmx', '.preset')

for ( $pat_idx = 0; $pat_idx -lt $fr_pats_list.count; $pat_idx++)
{
	$pat_dir = (-join($FreiburgFilesDirectory, $fr_pats_list[$pat_idx]))
	$pat_dir
	
	for ( $ext_idx = 0; $ext_idx -lt $to_delete_extensions.count; $ext_idx++) {

		$ext_str = (-join('*', $to_delete_extensions[$ext_idx]))
		$to_del_files = Get-ChildItem -Path $pat_dir -Filter $ext_str -Recurse -File -Name
		for ( $del_idx = 0; $del_idx -lt $to_delete_extensions.count; $del_idx++) {			
			$del_file_path = (-join($pat_dir, "\", $to_del_files[$del_idx]))
			#$del_file_path
			if (!$del_file_path.Contains(".lay") -and !$del_file_path.Contains(".dat") -and !$del_file_path.Contains(".csv")) {
				if (Test-Path $del_file_path) {
					$del_file_path
					if (!$target.PSIsContainer) {
						#$del_file_path
						Remove-Item $del_file_path -verbose
					} else {
						Read-Host -Prompt "This is a directory, Press Enter to continue"
					}
				} else {
					Read-Host -Prompt "Path doesn't exits, Press Enter to continue"
				}
			}
		}
	}
}

Read-Host -Prompt 'Press Enter to exit'
