# Starts the backend server from the correct folder and logs output
param(
    [string]$LogPath = ""
)

# Use repo root as base
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"

# Default log path inside backend if none provided
if (-not $LogPath -or $LogPath.Trim().Length -eq 0) {
    $LogPath = Join-Path $backend "server.log"
}

Write-Host "Starting backend from: $backend"
Write-Host "Logging to: $LogPath"

Push-Location $backend
try {
    npm start 2>&1 | Tee-Object -FilePath $LogPath
}
finally {
    Pop-Location
}