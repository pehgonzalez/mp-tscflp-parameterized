# run_mip_batch.ps1 - exact-baseline campaign (gurobi_port binary)
# on the author's Windows machine (i9-14900HX). Run with:
#   powershell -ExecutionPolicy Bypass -File run_mip_batch.ps1
#
# CLI of the companion binary (gurobi_port/src/main.cpp):
#   mptscfl.exe <instance> <lb_method> <time_limit> [mode] [seed] [threads]
#     lb_method 0 + mode "exact" = compact MIP (1)-(10)  -> labelled Mip
#     lb_method 1 + mode "exact" = branch-and-Benders-cut -> labelled Bd
#       (mode "exact" skips the two-steps heuristic warm start, so the runs
#        measure the pure exact methods; cuts = the disaggregated optimality cuts)
# The binary itself appends one summary row per run to logs\results.csv
# (header: datetime,instance,mode,method,seed,threads,time_limit_s,status,obj,
#  bound,gap,solver_time_s,total_wall_s,...,gurobi_version,gurobi_log).
# This script only sequences the runs; at the end it copies logs\results.csv
# to $OUT_DIR\mip_results.csv. Q2 pairs these rows with xp_results.csv; Q3
# correlates Mauri Mip times/gaps with results\mauri_kstar.csv (Spearman).
#
# Time limits (pre-registered): 600 s on the generated family
# (paired with the 60 s Xp runs; 10x budget for the baseline), 3600 s on the
# 100 Mauri instances (Q3 needs time-to-optimality where attainable; bonus:
# first-time-closed benchmark instances are a reportable result).
#
# ---- REQUIRED PATHS (adjust these four lines only) -------------------------
$MPTSCFL_EXE = "..\..\gurobi_port\build\Release\mptscfl.exe"
$KSTAR_DIR   = ".\data\instances_kstar"                    # generated family
$DATA_DIR    = "..\..\benchmark_data"
$OUT_DIR     = ".\results"
# -----------------------------------------------------------------------------
$TL_FAMILY = 600
$TL_MAURI  = 3600
$Seed      = 0      # Gurobi Seed (recorded per row by the binary)
$Threads   = 0      # 0 = Gurobi auto (record: i9-14900HX, 24 cores)
$Mode      = "exact"

if (-not (Test-Path $MPTSCFL_EXE)) { Write-Error "mptscfl.exe not found at $MPTSCFL_EXE"; exit 1 }
$MPTSCFL_EXE = (Resolve-Path $MPTSCFL_EXE).Path
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
$runs = @()   # (file, time_limit) pairs; methods {0,1} run for each
Get-ChildItem -Path $KSTAR_DIR -Filter "kstar_*.txt" | Sort-Object Name |
    ForEach-Object { $runs += ,@($_.FullName, $TL_FAMILY) }
if (Test-Path $DATA_DIR) {
    Get-ChildItem -Path $DATA_DIR -Filter "PSC*.txt" | Sort-Object Name |
        ForEach-Object { $runs += ,@($_.FullName, $TL_MAURI) }
} else {
    Write-Warning "Mauri dir not found: $DATA_DIR (skipping benchmark)"
}

# Resume support: logs\results.csv rows already present are not re-run.
$done = @{}
if (Test-Path "logs\results.csv") {
    Import-Csv "logs\results.csv" | ForEach-Object {
        $done[$_.instance + "|" + $_.method] = $true }
}

foreach ($r in $runs) {
    $file = $r[0]; $tl = $r[1]
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file)
    foreach ($method in 0, 1) {
        if ($done.ContainsKey($stem + "|" + $method)) {
            Write-Host ("[skip] " + $stem + " m" + $method); continue }
        Write-Host ("[m" + $method + " tl" + $tl + "] " + $stem)
        & $MPTSCFL_EXE $file $method $tl $Mode $Seed $Threads |
            Out-File -Encoding utf8 -Append (Join-Path $OUT_DIR "mip_console.log")
    }
}
Copy-Item "logs\results.csv" (Join-Path $OUT_DIR "mip_results.csv") -Force
Write-Host ("done -> " + (Join-Path $OUT_DIR "mip_results.csv"))
