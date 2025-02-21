<# $fr_pats_list = @(
'pat_FR_1073',
'pat_FR_1077_NoScalp',
'pat_FR_1084',
'pat_FR_1096',
'pat_FR_1125',
'pat_FR_1146_NoScalp',
'pat_FR_1150_NoScalp',
'pat_FR_13245_NoScalp',
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
'pat_FR_970'
) #>


#$fr_pats_list = @('pat_FR_1084','pat_FR_1096','pat_FR_253','pat_FR_970')
$fr_pats_list = @('pat_FR_273')
$fr_pats_list

$FreiburgFilesDirectory = "F:\FREIBURG_Simultaneous_OneHrFiles\"
for ( $pat_idx = 0; $pat_idx -lt $fr_pats_list.count; $pat_idx++)
{
	$pat_dir = (-join($FreiburgFilesDirectory, $fr_pats_list[$pat_idx]))
	$pat_dir
	
	$all_lay_files = Get-ChildItem -Path $pat_dir -Filter *.lay -Recurse -File -Name
	for ( $file_idx = 0; $file_idx -lt $all_lay_files.count; $file_idx++)
	{			
		$SourceFilePath =  (-join($pat_dir, '\', $all_lay_files[$file_idx]))
		$OutputFilePath = $SourceFilePath.Replace('.lay', '_ScalpSleepStages.csv')
		
		$trendsDoneFilePath = $SourceFilePath.Replace('.lay', '.mg2')
		if (Test-Path -Path $trendsDoneFilePath) {
			& "C:\Program Files (x86)\Persyst\Insight\PSCLI.exe" /SourceFile=$SourceFilePath /ExportCSV /Panel="ScalpSleepStages" /OutputFile=$OutputFilePath
		} else {
			Write-Host (-join("Trends for file:", $SourceFilePath, " not done yet"))
		}
		#$SourceFilePath
		#$OutputFilePath
	}
	"`n`n"
}

Read-Host -Prompt "Press Enter to exit"