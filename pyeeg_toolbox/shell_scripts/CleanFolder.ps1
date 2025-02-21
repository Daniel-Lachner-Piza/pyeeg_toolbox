$fr_pats_list = @(
'pat_FR_253',
'pat_FR_264',
'pat_FR_273',
'pat_FR_375',
'pat_FR_384',
'pat_FR_548',
'pat_FR_565',
'pat_FR_583',
'pat_FR_732_defekt',
'pat_FR_862',
'pat_FR_916',
'pat_FR_922',
'pat_FR_958',
'pat_FR_970',
'pat_FR_1073',
'pat_FR_1077_NoScalp',
'pat_FR_1084',
'pat_FR_1096',
'pat_FR_1125',
'pat_FR_1146_NoScalp',
'pat_FR_1150_NoScalp',
'pat_FR_13245_NoScalp'
)

$Pat_Name = 'pat_FR_565'
$fr_pats_list = @($Pat_Name)

$fr_pats_list

$FreiburgFilesDirectory = "F:\FREIBURG_Simultaneous_OneHrFiles\"
for ( $pat_idx = 0; $pat_idx -lt $fr_pats_list.count; $pat_idx++)
{
	$pat_dir = (-join($FreiburgFilesDirectory, $fr_pats_list[$pat_idx]))
	$pat_dir
	
	# Remove files in folder and subfolders
	$all_files = Get-ChildItem -Path $pat_dir -Filter * -Recurse -File -Name
	for ( $file_idx = 0; $file_idx -lt $all_files.count; $file_idx++)
	{	
		$filename = $all_files[$file_idx]
		$FilePath =  (-join($pat_dir, '\', $filename))

		$KeepFile = ($filename -Match ".lay" -or $filename -Match ".dat" -or $filename -Match ".csv" -or $filename -Match ".mat")		
		if (-Not $KeepFile ) {
			$FilePath
			Remove-Item $FilePath
		}
	}
	"`n`n"
	
	
	# Remove folders and subfolders
	$all_folders = Get-ChildItem -Path $pat_dir -Filter * -Recurse -Directory -Name
	for ( $file_idx = 0; $file_idx -lt $all_folders.count; $file_idx++)
	{	
		$foldername = $all_folders[$file_idx]
		$FolderPath =  (-join($pat_dir, '\', $foldername))		
		$FolderPath
		#Remove-Item $FolderPath -Force -Recurse -ErrorAction SilentlyContinue
		Remove-Item $FolderPath

	}
	"`n`n"
}

Read-Host -Prompt "Press Enter to exit"