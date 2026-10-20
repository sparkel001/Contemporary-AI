param([string]$DocumentPath = '')
$ErrorActionPreference = 'Stop'
if (-not $DocumentPath) {
    $DocumentPath = Join-Path (Split-Path $PSScriptRoot) 'report/实验二_搜索算法_实验报告.docx'
}
$inputFile = (Resolve-Path -LiteralPath $DocumentPath).Path
$pdfFile = [System.IO.Path]::ChangeExtension($inputFile, '.pdf')
$reportWord = New-Object -ComObject Word.Application
$reportDoc = $null
try {
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDoc = $reportWord.Documents.Open($inputFile, $false, $false)
    $reportDoc.EmbedTrueTypeFonts = $true
    $reportDoc.SaveSubsetFonts = $true
    $reportDoc.DoNotEmbedSystemFonts = $false
    $reportDoc.Fields.Update() | Out-Null
    $reportDoc.Repaginate()
    $reportDoc.Save()
    $pages = $reportDoc.ComputeStatistics(2)
    $reportDoc.ExportAsFixedFormat($pdfFile, 17)
    Write-Output "Exported PDF: $pdfFile; pages: $pages"
} finally {
    if ($null -ne $reportDoc) { $reportDoc.Close(0); [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDoc) | Out-Null }
    $reportWord.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportWord) | Out-Null
}
