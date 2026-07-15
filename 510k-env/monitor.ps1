# Monitor training progress
Write-Host "=== Training Monitor ==="
Write-Host "Time: $(Get-Date)"

# Check if training process is running
$procs = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($procs.Count -ge 2) {
    Write-Host "Status: RUNNING (PID: $($procs[-1].Id), CPU: $($procs[-1].TotalProcessorTime.TotalMinutes):.1f min)" -ForegroundColor Green
} else {
    Write-Host "Status: NOT RUNNING" -ForegroundColor Red
}

# Check model checkpoints
$models = Get-ChildItem -Path "models_selfplay" -Filter "*_final.zip" -Name -ErrorAction SilentlyContinue
if ($models) {
    Write-Host "Completed seeds:" -ForegroundColor Cyan
    $models | ForEach-Object { Write-Host "  $_" }
}

# Count total checkpoints for current seed
$ckpts = Get-ChildItem -Path "models_selfplay" -Filter "*.zip" -Name -ErrorAction SilentlyContinue
if ($ckpts) {
    $latest = ($ckpts | Sort-Object -Descending | Select-Object -First 1)
    $steps = [regex]::Match($latest, '(\d+)_steps').Groups[1].Value
    Write-Host "Current progress: $steps / 1000000 steps ($([math]::Round($steps/1000000*100,1))%)"
    Write-Host "Latest checkpoint: $latest"
    Write-Host "Total checkpoints: $($ckpts.Count)"
}

# Last log lines
$log = Get-Content -Path "train_all_out.log" -Tail 3 -ErrorAction SilentlyContinue
if ($log) {
    Write-Host "`nRecent log:" -ForegroundColor Yellow
    $log | ForEach-Object { Write-Host "  $_" }
}
