# Launches the Streamlit app using the project's Python 3.10 virtualenv.
# Usage:  .\run.ps1            (defaults to port 8600)
#         .\run.ps1 -Port 8700
param([int]$Port = 8600)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# C: drive is full, so keep all caches/browsers inside the project on F:.
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $PSScriptRoot ".pw-browsers"
# HuggingFace model cache (embedding model is downloaded here on first run).
$env:HF_HOME = Join-Path $PSScriptRoot ".hf-cache"

& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run app.py --server.port $Port
