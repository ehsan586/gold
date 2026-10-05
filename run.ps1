$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Safe wrapper: keep all quoting/process orchestration in bootstrap.py.
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if ($py) {
    & $py.Source -3 (Join-Path $PSScriptRoot 'bootstrap.py')
} else {
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $python) { throw 'Python 3.10+ was not found in PATH.' }
    & $python.Source (Join-Path $PSScriptRoot 'bootstrap.py')
}
exit $LASTEXITCODE
