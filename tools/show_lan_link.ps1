$ErrorActionPreference = "SilentlyContinue"

function Get-LanIps {
    $ips = @()
    Get-NetIPAddress -AddressFamily IPv4 | ForEach-Object {
        $ip = $_.IPAddress
        if ($ip -and $ip -notlike "127.*" -and $ip -notlike "169.254.*") {
            $ips += $ip
        }
    }
    if ($ips.Count -eq 0) {
        $line = (ipconfig | Select-String "IPv4" | Select-Object -First 1)
        if ($line -match ":\s*([\d.]+)") { $ips += $matches[1] }
    }
    return $ips | Select-Object -Unique
}

function Test-ServerBinding {
    $lines = netstat -ano | Select-String ":8000" | Select-String "LISTENING"
    foreach ($line in $lines) {
        if ($line -match "0\.0\.0\.0:8000") { return "lan" }
        if ($line -match "127\.0\.0\.1:8000") { return "local" }
    }
    return "down"
}

& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "open_firewall.ps1")

$bind = Test-ServerBinding
if ($bind -eq "down") {
    Write-Host ""
    Write-Host "  Server not running on port 8000." -ForegroundColor Red
    exit 1
}
if ($bind -eq "local") {
    Write-Host ""
    Write-Host "  WARNING: server only on 127.0.0.1 (not visible to others)." -ForegroundColor Red
    Write-Host "  Restart via start_lan_link.bat" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "           LAN LINK (same Wi-Fi only!)" -ForegroundColor Green
Write-Host ""

$ips = Get-LanIps
if ($ips.Count -eq 0) {
    Write-Host "  IP not found. Check Wi-Fi." -ForegroundColor Red
    exit 1
}

$primary = $ips[0]
foreach ($ip in $ips) {
    $url = "http://${ip}:8000/"
    Write-Host "           $url" -ForegroundColor Yellow
}
try { Set-Clipboard -Value "http://${primary}:8000/" } catch { }

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Friend must be on THE SAME Wi-Fi (not mobile internet)." -ForegroundColor White
Write-Host "  Link must start with http://192.168... NOT trycloudflare.com" -ForegroundColor White
Write-Host "  Error 530 = wrong link (Cloudflare). Use link above." -ForegroundColor Yellow
Write-Host "  Do NOT close server window." -ForegroundColor Gray
Write-Host ""

$dir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$linkFile = Join-Path $dir "artifacts\lan_link.txt"
Set-Content -Path $linkFile -Value "http://${primary}:8000/" -Encoding UTF8
