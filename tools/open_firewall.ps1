$ruleName = "ClimbAI"
$existing = netsh advfirewall firewall show rule name="$ruleName" 2>$null
if ($LASTEXITCODE -ne 0) {
    netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=8000 profile=any | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Firewall: port 8000 opened." -ForegroundColor Green
    } else {
        Write-Host "  Firewall: run as Administrator to open port 8000." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Firewall: port 8000 already allowed." -ForegroundColor Green
}
