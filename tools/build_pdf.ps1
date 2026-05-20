$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PdfDir = Join-Path $RepoRoot "pdf"

Push-Location $PdfDir
try {
    latexmk -pdf main.tex
    latexmk -c main.tex

    $Patterns = @(
        "*.aux", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk",
        "*.synctex.gz", "*.bbl", "*.blg", "*.nav", "*.snm"
    )

    Get-ChildItem -File | Where-Object {
        $name = $_.Name
        $Patterns | Where-Object { $name -like $_ }
    } | Remove-Item -Force
}
finally {
    Pop-Location
}
