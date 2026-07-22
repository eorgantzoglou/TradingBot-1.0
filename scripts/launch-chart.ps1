# Launches a Chromium browser with the Chrome DevTools Protocol port open and
# TradingView loaded, so the bot can read the chart.
#
#   npm run chart
#
# Uses a DEDICATED browser profile. This is not optional: Chrome refuses to
# open a debugging port when an instance is already running on the default
# profile -- it silently hands the URL to the existing window and no port is
# opened. A separate --user-data-dir gives us our own instance every time.

param(
    [int]$Port = 9222,
    [string]$Url = "https://www.tradingview.com/chart/"
)

$ErrorActionPreference = 'Stop'

$candidates = @(
    "$env:PROGRAMFILES\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
    "$env:PROGRAMFILES\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)
$browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) {
    Write-Error "No Chrome or Edge installation found. Install Chrome, or edit this script."
    exit 1
}

# Refuse to fight an existing listener on the port.
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port $Port is already serving CDP - reusing that browser." -ForegroundColor Yellow
    Write-Host "If it is not TradingView, close it first, then re-run this."
    exit 0
}

$profileDir = Join-Path $env:LOCALAPPDATA "TradingBot\cdp-profile"
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null

Write-Host "Browser : $browser"
Write-Host "Profile : $profileDir"
Write-Host "Port    : $Port"

Start-Process -FilePath $browser -ArgumentList @(
    "--remote-debugging-port=$Port",
    "--user-data-dir=`"$profileDir`"",
    "--no-first-run",
    "--no-default-browser-check",
    $Url
)

# Wait for the port to actually accept connections before declaring success.
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    try {
        $v = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/json/version" -TimeoutSec 2
        Write-Host ""
        Write-Host "CDP is live: $($v.Browser)" -ForegroundColor Green
        Write-Host "Log in to TradingView, open a chart, add your indicators,"
        Write-Host "then run:  npm run dry-run"
        exit 0
    } catch { }
}
Write-Error "Browser started but port $Port never opened. Is another browser instance blocking it?"
exit 1
