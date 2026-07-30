# run_pipeline.ps1 - COMPLETE and RESUMABLE experimental pipeline (MP-TSCFLP).
#
# Resumes from logs\results.csv (skips every (instance|method) pair already
# recorded) and runs everything still missing for the paper, in priority order,
# printing [X/Tot] and "remaining N" on screen so you know at any moment how
# many runs are left. Safe against Ctrl-C / shutdown: only completed runs enter
# results.csv, so on restart it continues exactly where it stopped (a run
# interrupted mid-way is simply redone). Sleep/hibernate is fine: the Gurobi
# budget counts active time, not wall clock.
#
# PHASES (in this order):
#   A. Mauri  Mip (method 0, 3600s) -> Q3 (k* vs solver effort)       100 runs
#   B. Family Mip (method 0, 600s)  -> Q2 (Mip vs Xp)                 400 runs
#   C. Family Bd  (method 1, 600s)  -> Q2 (Bd closure)                400 runs
#   Pipeline total: 900 runs. Already completed ones are skipped.
#
# Approximate ETA (measured on the runs already done):
#   A ~1h/instance (almost everything hits the limit)  -> ~3 days for the 76 remaining
#   B ~0.2s/instance (the 78 done: all OPTIMAL)        -> minutes
#   C ~575s/instance (90% hit the 600s)                -> ~2 days for the 323 remaining
#
# The Bd block is refinement: the runs so far already show that Bd closes only
# a minority, so you MAY Ctrl-C at the end of the Mip block without losing the essentials.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run_pipeline.ps1

$ErrorActionPreference = "Stop"

# ---- PATHS (adjust only if you move the folders) --------------------------
$MPTSCFL_EXE = ".\gurobi_port\build\Release\mptscfl.exe"
$MAURI_DIR   = "..\..\benchmark_data"
$FAMILY_DIR  = ".\data\instances_kstar"
$OUT_DIR     = ".\results"
$LOG_CSV     = ".\logs\results.csv"
$CONSOLE_LOG = Join-Path $OUT_DIR "pipeline_console.log"
# ---- PROTOCOL (do not change mid-campaign) --------------------------------
$TL_MAURI  = 3600
$TL_FAMILY = 600
$Mode = "exact"; $Seed = 0; $Threads = 0
# ---------------------------------------------------------------------------

if (-not (Test-Path $MPTSCFL_EXE)) { Write-Error "binary not found: $MPTSCFL_EXE"; exit 1 }
$MPTSCFL_EXE = (Resolve-Path $MPTSCFL_EXE).Path
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LOG_CSV) | Out-Null

# ---- build the ordered plan: list of runs (file, method, tl, phase) -------
$plan = New-Object System.Collections.ArrayList
function Add-Phase($dir, $filter, $method, $tl, $phase) {
    if (-not (Test-Path $dir)) { Write-Warning "missing folder: $dir  (phase $phase skipped)"; return }
    Get-ChildItem -Path $dir -Filter $filter | Sort-Object Name | ForEach-Object {
        [void]$plan.Add([pscustomobject]@{ File=$_.FullName; Method=$method; TL=$tl; Phase=$phase })
    }
}
Add-Phase $MAURI_DIR  "PSC*.txt"    0 $TL_MAURI  "A Mauri-Mip"   # Q3
Add-Phase $FAMILY_DIR "kstar_*.txt" 0 $TL_FAMILY "B Fam-Mip"     # Q2 Mip
Add-Phase $FAMILY_DIR "kstar_*.txt" 1 $TL_FAMILY "C Fam-Bd"      # Q2 Bd
$Tot = $plan.Count

# ---- already-completed set (resume) ---------------------------------------
$done = @{}
if (Test-Path $LOG_CSV) {
    Import-Csv $LOG_CSV | ForEach-Object { $done[$_.instance + "|" + $_.method] = $true }
}
$doneCount = 0
$pendByPhase = [ordered]@{ "A Mauri-Mip" = 0; "B Fam-Mip" = 0; "C Fam-Bd" = 0 }
foreach ($p in $plan) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($p.File)
    if ($done.ContainsKey($stem + "|" + $p.Method)) { $doneCount++ }
    else { $pendByPhase[$p.Phase] = 1 + [int]$pendByPhase[$p.Phase] }
}
$pending = $Tot - $doneCount

Write-Host ""
Write-Host "==================================================================="
Write-Host (" PIPELINE MP-TSCFLP  -  {0} runs in total" -f $Tot)
Write-Host ("   completed: {0}/{1}    |    remaining: {2}" -f $doneCount, $Tot, $pending)
Write-Host ("   remaining per phase:  A Mauri-Mip {0}  |  B Fam-Mip {1}  |  C Fam-Bd {2}" -f `
    $pendByPhase["A Mauri-Mip"], $pendByPhase["B Fam-Mip"], $pendByPhase["C Fam-Bd"])
Write-Host "   safe to shut down / Ctrl-C: resumes exactly from here"
Write-Host "   detailed solver output in results\pipeline_console.log"
Write-Host "==================================================================="
Write-Host ""

# ---- main loop --------------------------------------------------------------
# "completed" and "remaining" always sum to Tot (900): completed goes up after
# each run, remaining = Tot - completed (includes the one starting right now).
$completed = $doneCount
$ranThisSession = 0
foreach ($p in $plan) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($p.File)
    if ($done.ContainsKey($stem + "|" + $p.Method)) { continue }   # already done -> skip
    $left  = $Tot - $completed
    $stamp = (Get-Date).ToString("HH:mm:ss")
    Write-Host ("completed {0}/{1}  |  remaining {2}  ||  {3}  {4}  m{5}  tl{6}s  (start {7})" -f `
        $completed, $Tot, $left, $p.Phase, $stem, $p.Method, $p.TL, $stamp)
    & $MPTSCFL_EXE $p.File $p.Method $p.TL $Mode $Seed $Threads 2>&1 |
        Tee-Object -FilePath $CONSOLE_LOG -Append | Out-Null
    $completed++
    $ranThisSession++
}

# ---- consolidate and summarize ----------------------------------------------
Copy-Item $LOG_CSV (Join-Path $OUT_DIR "mip_results.csv") -Force
$rows     = Import-Csv $LOG_CSV
$mauriM0  = @($rows | Where-Object { $_.instance -like "PSC*"  -and $_.method -eq "0" })
$famM0    = @($rows | Where-Object { $_.instance -notlike "PSC*" -and $_.method -eq "0" })
$famM1    = @($rows | Where-Object { $_.instance -notlike "PSC*" -and $_.method -eq "1" })
$mauriOpt = @($mauriM0 | Where-Object { $_.status -eq "OPTIMAL" }).Count

Write-Host ""
Write-Host "==================================================================="
Write-Host ("  session ended: {0} runs executed" -f $ranThisSession)
Write-Host ("  Mauri-Mip : {0}/100   (proven OPTIMAL: {1})" -f $mauriM0.Count, $mauriOpt)
Write-Host ("  Fam-Mip   : {0}/400" -f $famM0.Count)
Write-Host ("  Fam-Bd    : {0}/400" -f $famM1.Count)
Write-Host ("  consolidated data -> {0}" -f (Join-Path $OUT_DIR "mip_results.csv"))
Write-Host "  When it finishes (or when you stop it), let me know: I run the"
Write-Host "  finalization (Q3/Q2, figures, paper recompile) from that CSV."
Write-Host "==================================================================="
