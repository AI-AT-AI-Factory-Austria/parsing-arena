<# Start the installed production frontend and backend in hidden processes. #>
param([ValidateSet('windows-amd', 'cpu')][string]$Profile)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$profileFile = Join-Path $repo 'data/installation.json'
if (-not $Profile) {
  if (-not (Test-Path -LiteralPath $profileFile)) { throw 'Run scripts/setup-windows.ps1 first, or supply -Profile.' }
  $Profile = (Get-Content -Raw -LiteralPath $profileFile | ConvertFrom-Json).profile
}
if ($Profile -notin @('windows-amd', 'cpu')) { throw 'Invalid installation profile.' }
foreach ($port in @(3000, 8000)) {
  if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $port is already in use. Stop the existing app before starting another instance."
  }
}
$python = Join-Path $repo 'backend/.venv/Scripts/python.exe'
$node = (Get-Command node -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath (Join-Path $repo 'frontend/.next/BUILD_ID'))) { throw 'Run setup first to build the frontend.' }
$logDir = Join-Path $repo 'data/logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$previousDevice = $env:LOCAL_DEVICE
$previousProfile = $env:RUNTIME_PROFILE
try {
  $env:LOCAL_DEVICE = if ($Profile -eq 'cpu') { 'cpu' } else { 'auto' }
  $env:RUNTIME_PROFILE = if ($Profile -eq 'cpu') { 'auto' } else { 'windows-amd' }
  $backend = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000','--timeout-keep-alive','65' -WorkingDirectory (Join-Path $repo 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$stamp-backend.out.log") -RedirectStandardError (Join-Path $logDir "$stamp-backend.err.log") -PassThru
  $frontend = Start-Process -FilePath $node -ArgumentList 'node_modules/next/dist/bin/next','start','--hostname','127.0.0.1','--port','3000' -WorkingDirectory (Join-Path $repo 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$stamp-frontend.out.log") -RedirectStandardError (Join-Path $logDir "$stamp-frontend.err.log") -PassThru
  Write-Output "Starting Parsing Arena at http://localhost:3000 (backend PID $($backend.Id), frontend PID $($frontend.Id)). Logs: $logDir"
} finally {
  $env:LOCAL_DEVICE = $previousDevice
  $env:RUNTIME_PROFILE = $previousProfile
}
