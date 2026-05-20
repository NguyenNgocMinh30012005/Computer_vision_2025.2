param(
    [string]$Message = "chore(repo): organize experiment repository"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$LatexIntermediatePatterns = @(
    "*.aux", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk",
    "*.synctex.gz", "*.bbl", "*.blg", "*.nav", "*.snm"
)
$LatexIntermediates = Get-ChildItem -Path "pdf" -File | Where-Object {
    $name = $_.Name
    $LatexIntermediatePatterns | Where-Object { $name -like $_ }
}

if ($LatexIntermediates.Count -gt 0) {
    Write-Error "LaTeX intermediate files remain in pdf/. Run the documented latexmk cleanup first."
}

$SafePaths = @(
    ".gitattributes",
    ".gitignore",
    "README.md",
    "LICENSE",
    "agents.md",
    "docs",
    "notebooks",
    "pdf",
    "scripts",
    "tools"
)

git add -u -- .
git add -- $SafePaths

$Staged = git diff --cached --name-only
if (-not $Staged) {
    Write-Host "No staged changes to commit."
} else {
    git commit -m $Message
}

git push origin main
