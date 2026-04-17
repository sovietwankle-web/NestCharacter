$ErrorActionPreference = "Stop"

# Simple smoke run: only verify pipeline can execute.
if ($PSScriptRoot) {
  Set-Location $PSScriptRoot
}

$OutDir = "trial_quick_run"

python generate_dataset_copypaste.py `
  --out-dir $OutDir `
  --layer-min 5 `
  --layer-max 5 `
  --samples-per-class 1 `
  --canvas-size 4200 `
  --base-large-font-size 180 `
  --wrap-after 8 `
  --seed 123 `
  --export-debug-json `
  --debug-json-name compare_report.json `
  --export-box-overlay `
  --box-draw-step 8 `
  --box-outline-width 1

if ($LASTEXITCODE -ne 0) {
  throw "quick trial run failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Quick trial run finished."
Write-Host "Output dir: $OutDir"
Write-Host "Check files:"
Write-Host "  $OutDir\\metadata.json"
Write-Host "  $OutDir\\compare_report.json"
