<# $SD4FilesDirectory = "F:\Temp2\"
$AllSD4Files = Get-ChildItem -Path $SD4FilesDirectory -Filter *.sd4 -Recurse -File -Name
(-join("Nr. SD4 Files:", $AllSD4Files.count))
for ( $index = 0; $index -lt $AllSD4Files.count; $index++)
{
	$FilePath = (-join($SD4FilesDirectory, $AllSD4Files[$index]))
	& 'C:\Program Files (x86)\Persyst\Insight\ExportSD4_p14.exe' $FilePath
	$Progress = 100*(($index+1)/$AllSD4Files.count)
	(-join("Progress:", $Progress, "%"))
	"`n`n"
}
Read-Host -Prompt "Press Enter to exit" #>



#$fr_pats_list = @('pat_FR_1084','pat_FR_1096','pat_FR_253','pat_FR_970')
$fr_pats_list = @('pat_FR_384')
$fr_pats_list

$FreiburgFilesDirectory = "F:\FREIBURG_Simultaneous_OneHrFiles\"
for ( $pat_idx = 0; $pat_idx -lt $fr_pats_list.count; $pat_idx++)
{
	$pat_dir = (-join($FreiburgFilesDirectory, $fr_pats_list[$pat_idx]))
	$pat_dir
	
	$all_sd4_files = Get-ChildItem -Path $pat_dir -Filter *.sd4 -Recurse -File -Name
	(-join("Nr. SD4 Files:", $all_sd4_files.count))
	for ( $file_idx = 0; $file_idx -lt $all_sd4_files.count; $file_idx++)
	{			
		$FilePath = (-join($pat_dir, '\', $all_sd4_files[$file_idx]))
		$FilePath
		
		& 'C:\Program Files (x86)\Persyst\Insight\ExportSD4_p14.exe' $FilePath
		$Progress = 100*(($file_idx+1)/$all_sd4_files.count)
		(-join("Progress:", $Progress, "%"))
		"`n`n"
	}
}
Read-Host -Prompt "Press Enter to exit"