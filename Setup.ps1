# Run from the VS Code terminal: powershell -ExecutionPolicy Bypass -File .\Setup.ps1
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed. Install Python 3.11 or newer first.' }
}
& '.venv/Scripts/python.exe' -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python version' }
& '.venv/Scripts/python.exe' -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and interpreter version.' }
Write-Output 'Setup complete. In VS Code select .venv/Scripts/python.exe, then press F5.'
Write-Output 'Terminal: .\.venv\Scripts\python.exe run.py play'

