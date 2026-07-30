# run_xp_batch.ps1 - Xp campaign (branch-and-bound solver) on the author's
# Windows machine (i9-14900HX). Run with:
#   powershell -ExecutionPolicy Bypass -File run_xp_batch.ps1
#
# Runs xp.exe (60 s limit, mode k=-1 = plain optimize) over
#   (1) the generated k*-knob family (instances_kstar, 400 files + MANIFEST.csv)
#   (2) the 100 Mauri benchmark instances (PSC*.txt)
# and APPENDS each solver one-line record (the output contract of
# code/README.md) to $OUT_DIR\xp_results.csv. Resumable: instances already
# recorded in the CSV are skipped, so the script can be re-run after an
# interruption. Everything except the time= field is deterministic.
#
# ---- REQUIRED PATHS (adjust these four lines only) -------------------------
$XP_EXE    = ".\build\xp.exe"            # build: g++ -std=c++20 -O2 -o build\xp.exe src\solver_xp.cpp src\main_xp.cpp
$KSTAR_DIR = ".\data\instances_kstar"    # generated family (MANIFEST.csv alongside)
$DATA_DIR  = "..\..\benchmark_data"
$OUT_DIR   = ".\results"
# -----------------------------------------------------------------------------
$TimeLimit = 60        # seconds per instance (pre-registered Q1/Q2 budget)
$K         = -1        # -1 = plain optimize (k = n)

if (-not (Test-Path $XP_EXE)) { Write-Error "xp.exe not found at $XP_EXE"; exit 1 }
$XP_EXE = (Resolve-Path $XP_EXE).Path
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
$OutFile = Join-Path $OUT_DIR "xp_results.csv"
if (-not (Test-Path $OutFile)) { "line" | Out-File -Encoding utf8 $OutFile }
$done = @{}
Get-Content $OutFile | ForEach-Object {
    if ($_ -match "instance=(\S+)") { $done[$Matches[1]] = $true }
}

$instances = @()
$instances += Get-ChildItem -Path $KSTAR_DIR -Filter "kstar_*.txt" | Sort-Object Name
if (Test-Path $DATA_DIR) {
    $instances += Get-ChildItem -Path $DATA_DIR -Filter "PSC*.txt" | Sort-Object Name
} else {
    Write-Warning "Mauri dir not found: $DATA_DIR (skipping benchmark)"
}

("# run_xp_batch " + (Get-Date -Format o) + " host=" + $env:COMPUTERNAME +
 " time_limit=" + $TimeLimit + " k=" + $K) | Out-File -Encoding utf8 -Append $OutFile
foreach ($inst in $instances) {
    if ($done.ContainsKey($inst.Name)) { Write-Host ("[skip] " + $inst.Name); continue }
    Write-Host ("[xp] " + $inst.Name)
    $line = & $XP_EXE $inst.FullName $TimeLimit $K
    $line | Out-File -Encoding utf8 -Append $OutFile
}
Write-Host "done -> $OutFile"
