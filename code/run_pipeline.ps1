# run_pipeline.ps1 - Pipeline experimental COMPLETO e RESUMIVEL (MP-TSCFLP).
#
# Retoma de logs\results.csv (pula todo par (instancia|metodo) ja registrado) e
# roda tudo que ainda falta para o paper, em ordem de prioridade, mostrando na
# tela [X/Tot] e "faltam N" para voce saber a qualquer momento quantas execucoes
# restam. Seguro para Ctrl-C / desligar: so runs concluidos entram no results.csv,
# entao ao reiniciar ele continua exatamente de onde parou (o run interrompido no
# meio e simplesmente refeito). Dormir/hibernar tudo bem: o orcamento do Gurobi
# conta tempo ativo, nao relogio de parede.
#
# FASES (nesta ordem):
#   A. Mauri  Mip (metodo 0, 3600s) -> Q3 (k* vs esforco do solver)   100 runs
#   B. Familia Mip (metodo 0, 600s) -> Q2 (Mip vs Xp)                 400 runs
#   C. Familia Bd  (metodo 1, 600s) -> Q2 (fechamento do Bd)          400 runs
#   Total do pipeline: 900 runs. Os ja concluidos sao pulados.
#
# ETA aproximado (medido nos runs ja feitos):
#   A ~1h/instancia (quase tudo estoura o limite)      -> ~3 dias para as 76 que faltam
#   B ~0.2s/instancia (as 78 feitas: todas OPTIMAL)    -> minutos
#   C ~575s/instancia (90% batem os 600s)              -> ~2 dias para as 323 que faltam
#
# O bloco Bd e refinamento: as execucoes ja evidenciam que o Bd fecha so uma
# minoria, entao PODE-se dar Ctrl-C ao fim do bloco Mip sem perder o essencial.
#
# Uso:  powershell -ExecutionPolicy Bypass -File run_pipeline.ps1

$ErrorActionPreference = "Stop"

# ---- CAMINHOS (ajuste apenas se mover as pastas) --------------------------
$MPTSCFL_EXE = ".\gurobi_port\build\Release\mptscfl.exe"
$MAURI_DIR   = "..\..\benchmark_data"
$FAMILY_DIR  = ".\data\instances_kstar"
$OUT_DIR     = ".\results"
$LOG_CSV     = ".\logs\results.csv"
$CONSOLE_LOG = Join-Path $OUT_DIR "pipeline_console.log"
# ---- PROTOCOLO (nao mude no meio da campanha) -----------------------------
$TL_MAURI  = 3600
$TL_FAMILY = 600
$Mode = "exact"; $Seed = 0; $Threads = 0
# ---------------------------------------------------------------------------

if (-not (Test-Path $MPTSCFL_EXE)) { Write-Error "binario nao encontrado: $MPTSCFL_EXE"; exit 1 }
$MPTSCFL_EXE = (Resolve-Path $MPTSCFL_EXE).Path
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LOG_CSV) | Out-Null

# ---- monta o plano ordenado: lista de runs (arquivo, metodo, tl, fase) ----
$plan = New-Object System.Collections.ArrayList
function Add-Phase($dir, $filter, $method, $tl, $phase) {
    if (-not (Test-Path $dir)) { Write-Warning "pasta ausente: $dir  (fase $phase pulada)"; return }
    Get-ChildItem -Path $dir -Filter $filter | Sort-Object Name | ForEach-Object {
        [void]$plan.Add([pscustomobject]@{ File=$_.FullName; Method=$method; TL=$tl; Phase=$phase })
    }
}
Add-Phase $MAURI_DIR  "PSC*.txt"    0 $TL_MAURI  "A Mauri-Mip"   # Q3
Add-Phase $FAMILY_DIR "kstar_*.txt" 0 $TL_FAMILY "B Fam-Mip"     # Q2 Mip
Add-Phase $FAMILY_DIR "kstar_*.txt" 1 $TL_FAMILY "C Fam-Bd"      # Q2 Bd
$Tot = $plan.Count

# ---- conjunto ja concluido (resume) ---------------------------------------
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
Write-Host (" PIPELINE MP-TSCFLP  -  {0} execucoes no total" -f $Tot)
Write-Host ("   concluidas: {0}/{1}    |    faltam: {2}" -f $doneCount, $Tot, $pending)
Write-Host ("   faltam por fase:  A Mauri-Mip {0}  |  B Fam-Mip {1}  |  C Fam-Bd {2}" -f `
    $pendByPhase["A Mauri-Mip"], $pendByPhase["B Fam-Mip"], $pendByPhase["C Fam-Bd"])
Write-Host "   seguro para desligar / Ctrl-C: retoma exatamente daqui"
Write-Host "   saida detalhada do solver em results\pipeline_console.log"
Write-Host "==================================================================="
Write-Host ""

# ---- laco principal --------------------------------------------------------
# "concluidas" e "faltam" sempre somam Tot (900): concluidas sobe a cada run,
# faltam = Tot - concluidas (inclui a que esta iniciando agora).
$completed = $doneCount
$ranThisSession = 0
foreach ($p in $plan) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($p.File)
    if ($done.ContainsKey($stem + "|" + $p.Method)) { continue }   # ja feito -> pula
    $left  = $Tot - $completed
    $stamp = (Get-Date).ToString("HH:mm:ss")
    Write-Host ("concluidas {0}/{1}  |  faltam {2}  ||  {3}  {4}  m{5}  tl{6}s  (inicio {7})" -f `
        $completed, $Tot, $left, $p.Phase, $stem, $p.Method, $p.TL, $stamp)
    & $MPTSCFL_EXE $p.File $p.Method $p.TL $Mode $Seed $Threads 2>&1 |
        Tee-Object -FilePath $CONSOLE_LOG -Append | Out-Null
    $completed++
    $ranThisSession++
}

# ---- consolida e resume ----------------------------------------------------
Copy-Item $LOG_CSV (Join-Path $OUT_DIR "mip_results.csv") -Force
$rows     = Import-Csv $LOG_CSV
$mauriM0  = @($rows | Where-Object { $_.instance -like "PSC*"  -and $_.method -eq "0" })
$famM0    = @($rows | Where-Object { $_.instance -notlike "PSC*" -and $_.method -eq "0" })
$famM1    = @($rows | Where-Object { $_.instance -notlike "PSC*" -and $_.method -eq "1" })
$mauriOpt = @($mauriM0 | Where-Object { $_.status -eq "OPTIMAL" }).Count

Write-Host ""
Write-Host "==================================================================="
Write-Host ("  sessao encerrada: {0} execucoes rodadas" -f $ranThisSession)
Write-Host ("  Mauri-Mip : {0}/100   (OPTIMAL provados: {1})" -f $mauriM0.Count, $mauriOpt)
Write-Host ("  Fam-Mip   : {0}/400" -f $famM0.Count)
Write-Host ("  Fam-Bd    : {0}/400" -f $famM1.Count)
Write-Host ("  dados consolidados -> {0}" -f (Join-Path $OUT_DIR "mip_results.csv"))
Write-Host "  Ao terminar (ou quando parar), me avise: rodo a finalizacao"
Write-Host "  (Q3/Q2, figuras, recompilar o paper) a partir desse CSV."
Write-Host "==================================================================="
