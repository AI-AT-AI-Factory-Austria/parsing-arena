<#
.SYNOPSIS
  Prepare the complete native Windows/AMD or CPU environment without changing drivers.
  Requires signed Python 3.13 (py launcher), Node.js 22 LTS and uv.
#>
param(
  [ValidateSet('windows-amd', 'cpu')][string]$Profile = 'windows-amd',
  [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
function Run-Step([string]$Program, [string[]]$Arguments) {
  & $Program @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Fix the error above and rerun setup." }
}
foreach ($program in @('py', 'uv', 'node', 'npm.cmd')) {
  if (-not (Get-Command $program -ErrorAction SilentlyContinue)) { throw "Install $program first; see README > Windows / AMD." }
}
Run-Step 'py' @('-3.13', '--version')
Run-Step 'node' @('-e', "if (Number(process.versions.node.split('.')[0]) < 22) process.exit(1); console.log('Node', process.versions.node)")
if ($CheckOnly) { Write-Output 'Prerequisites found. No environments or settings changed.'; return }
Push-Location $repo
try {
  $python = Join-Path $repo 'backend/.venv/Scripts/python.exe'
  if (-not (Test-Path -LiteralPath $python)) { Run-Step 'py' @('-3.13', '-m', 'venv', 'backend/.venv') }
  Run-Step 'uv' @('sync', '--project', 'backend', '--locked', '--extra', 'local', '--inexact', '--python', $python)
  $index = if ($Profile -eq 'windows-amd') { 'https://repo.amd.com/rocm/whl/gfx1151/' } else { 'https://download.pytorch.org/whl/cpu' }
  Run-Step 'uv' @('--no-config', 'pip', 'install', '--python', $python, '--index-url', $index, '--reinstall-package', 'torch', '--reinstall-package', 'torchvision', 'torch==2.11.0', 'torchvision==0.26.0')
  $markerPython = Join-Path $repo 'backend/runtimes/marker/.venv/Scripts/python.exe'
  if (-not (Test-Path -LiteralPath $markerPython)) { Run-Step 'py' @('-3.13', '-m', 'venv', 'backend/runtimes/marker/.venv') }
  Run-Step 'uv' @('sync', '--project', 'backend/runtimes/marker', '--locked', '--python', $markerPython)
  Run-Step $python @('backend/scripts/install-preset-tools.py')
  Run-Step 'uv' @('run', '--python', $python, '--no-project', '--script', 'backend/scripts/install-skyvision.py')
  Run-Step 'uv' @('run', '--python', $python, '--no-project', '--script', 'backend/scripts/install-unlimited.py')
  Run-Step $python @('-m', 'playwright', 'install', 'chromium')
  if (-not (Get-Command tesseract -ErrorAction SilentlyContinue) -and -not (Test-Path 'C:\Program Files\Tesseract-OCR\tesseract.exe')) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
      Run-Step 'winget' @('install', '--id', 'UB-Mannheim.TesseractOCR', '--exact', '--silent')
    } else { throw 'Install Tesseract (README), then rerun setup.' }
  }
  if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
  Push-Location frontend
  try { Run-Step 'npm.cmd' @('ci'); Run-Step 'npm.cmd' @('run', 'build') }
  finally { Pop-Location }
  New-Item -ItemType Directory -Force -Path 'data' | Out-Null
  @{ profile = $Profile } | ConvertTo-Json | Set-Content -Encoding UTF8 'data/installation.json'
  Write-Output 'Setup complete. Run: .\scripts\start-windows.ps1'
} finally { Pop-Location }
