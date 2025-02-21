
<# $AllLayFiles = @("BARCLAY~ Aaron_f0c9c7eb-03de-41f1-a2c6-882717154ce6.lay",
"BRIGGSRIGIO~ S_3a8c8d43-bc55-4f4d-be14-40386df9aa3c.lay",
"CONTRERAS~ Hol_98d6f210-e07d-41dc-b04d-af63572570c0.lay",
"DEDRICK~ Olivi_c9f20569-fc7a-4c73-8f5f-fc71cfb6a1cf.lay",
"GOBLELIDSTONE~_e70cbf40-f8db-4212-a6d7-49bfb9e90804.lay",
"MABIOR~ Diing_1f5d1111-7f80-48ca-8b80-02cad3635657.lay",
"MASRI~ Malik_a0d5a33c-c6b9-4523-863a-f908f7ade054.lay",
"WICKHORST~ Jos_9fc045cf-a9d6-486a-9489-c51cf8044d48.lay",
"WILSON~ Jack_f7d8cb8c-685e-4ac5-83dd-ffeddb73130d.lay",
"WOOD~ Dawson_4137e0bc-a3b5-4572-95dc-937dd2c8ae56.lay"
) #>

$LayFilesDirectory = "C:\Users\HFO\Documents\Persyst_Project\Spike_Annotation_Files___Julia_Jacobs\"
$AllLayFiles = Get-ChildItem -Path $LayFilesDirectory -Filter *.lay -Recurse -File -Name
"`n`n`n"

(-join("Nr. Files to annotate", ": ", $AllLayFiles.count))
"Spike annotations in each of the EEG Files:`n"

for ( $index = 0; $index -lt $AllLayFiles.count; $index++)
{
	$FilePath = (-join($LayFilesDirectory, $AllLayFiles[$index]))
	$FileContent = Get-Content $FilePath
	$Matches = Select-String -InputObject $FileContent -Pattern "@Spike" -AllMatches	
	$OutStr = (-join($AllLayFiles[$index], ": ", $Matches.Matches.Count))
	$OutStr	
}

"`n`n`n"

Read-Host -Prompt "Press Enter to exit"