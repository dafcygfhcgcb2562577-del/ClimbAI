$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Cloudflared = Join-Path $Root "tools\cloudflared.exe"
$LinkFile = Join-Path $Root "artifacts\public_link.txt"

if (-not (Test-Path $Cloudflared)) {
    Write-Host "cloudflared.exe not found: $Cloudflared" -ForegroundColor Red
    exit 1
}

function Show-PublicLink {
    param([string]$Url)
    $dir = Split-Path $LinkFile -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -Path $LinkFile -Value $Url -Encoding UTF8
    try { Set-Clipboard -Value $Url } catch { }

    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "           PUBLIC LINK / PUBLICHNAYA SSYLKA" -ForegroundColor Green
    Write-Host ""
    Write-Host "           $Url" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Link copied to clipboard." -ForegroundColor Gray
    Write-Host "  Open it on another PC or phone." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  DO NOT CLOSE THIS WINDOW during demo!" -ForegroundColor White
    Write-Host "  If you close it -> error 530, link dies." -ForegroundColor White
    Write-Host "  Ctrl+C to stop." -ForegroundColor Gray
    Write-Host ""
}

function Wait-ForServer {
    param([int]$TimeoutSec = 45)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 1
        Write-Host "  Waiting for server http://127.0.0.1:8000 ..." -ForegroundColor DarkGray
    }
    return $false
}

if (-not (Wait-ForServer)) {
    Write-Host ""
    Write-Host "  Server did not start on port 8000." -ForegroundColor Red
    Write-Host "  Run run_all_in_one.bat and check errors." -ForegroundColor Red
    Write-Host ""
    exit 1
}

Write-Host "  Server ready." -ForegroundColor Green
Write-Host "  Creating public link..." -ForegroundColor Gray
Write-Host ""

$urlShown = $false
$currentUrl = ""
$lastAlive = Get-Date

& $Cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000 2>&1 | ForEach-Object {
    $line = "$_"

    if ($line -match "(https://[a-z0-9-]+\.trycloudflare\.com)") {
        if (-not $urlShown) {
            $script:currentUrl = $matches[1]
            Show-PublicLink -Url $currentUrl
            $urlShown = $true
        }
        return
    }

    if ($line -match "Connection terminated|Serve tunnel error|unregistered|context canceled") {
        Write-Host ""
        Write-Host "  !!! TUNNEL DISCONNECTED !!!" -ForegroundColor Red
        Write-Host "  Link no longer works (error 530)." -ForegroundColor Red
        Write-Host "  Restart start_public_link.bat and use NEW link." -ForegroundColor Yellow
        Write-Host ""
        return
    }

    if ($line -match "\sERR\s") {
        Write-Host "  $line" -ForegroundColor DarkRed
        return
    }

    if ($line -match "Registered tunnel connection") {
        $script:lastAlive = Get-Date
        if ($urlShown) {
            $ts = Get-Date -Format "HH:mm:ss"
            Write-Host "  $ts  Tunnel active." -ForegroundColor DarkGreen
        }
        return
    }
}

if (-not $urlShown) {
    Write-Host "No link received. Check internet and try again." -ForegroundColor Red
    exit 1
}
