# Real run of setup.ps1 on a Windows PC with an NVIDIA card, into a folder with spaces and non-ASCII letters.
# Usage (from the unpacked source folder): powershell -File installer\test_setup.ps1 -Uv <path to uv.exe>
param([Parameter(Mandatory = $true)][string]$Uv, [string]$Root = 'C:\aivibe-build\t\Ünï côde')
$ErrorActionPreference = 'Stop'
$src = Split-Path $PSScriptRoot
function Check([bool]$ok, [string]$what) { if (-not $ok) { Write-Output "FAIL: $what"; exit 1 } ; Write-Output "ok: $what" }

Remove-Item -Recurse -Force -LiteralPath $Root -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Root | Out-Null
Copy-Item -LiteralPath $Uv, "$src\installer\setup.ps1", "$src\installer\requirements.lock.txt", "$src\installer\python.version" -Destination $Root
$log = Join-Path $Root 'setup-test.log'
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
$flavour = if (-not $drv) { 'cpu' } elseif ([int]($drv.Split('.')[0]) -ge 580) { 'cu130' } else { 'cu126' }
Write-Output "driver '$drv' -> $flavour"

$t = Measure-Command { & powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour $flavour -Log $log }
Check ($LASTEXITCODE -eq 0) "first run exit 0 ($([int]$t.TotalMinutes) min)"
Check (Test-Path -LiteralPath "$Root\venv\Scripts\pythonw.exe") 'pythonw.exe exists'
$cuda = & "$Root\venv\Scripts\python.exe" -c "import torch; print('cuda', torch.cuda.is_available())"
Check ($flavour -eq 'cpu' -or $cuda -match 'cuda True') "torch sees the card ($cuda)"

$t2 = Measure-Command { & powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour auto -Log $log }
Check ($LASTEXITCODE -eq 0 -and $t2.TotalSeconds -lt 60) "second run skips ($([int]$t2.TotalSeconds) s)"
Check ((Get-Content -LiteralPath "$Root\torch.txt" -Raw).Trim() -eq $flavour) 'auto reads the saved flavour'
# Repair: break the environment, then -Force must rebuild it
Remove-Item -LiteralPath "$Root\venv\Lib\site-packages\diffusers\__init__.py"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour auto -Log $log | Out-Null
Check (-not (Test-Path -LiteralPath "$Root\venv\Lib\site-packages\diffusers\__init__.py")) 'without -Force a matching stamp skips'
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour auto -Force -Log $log | Out-Null
Check ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath "$Root\venv\Lib\site-packages\diffusers\__init__.py")) 'Repair (-Force) rebuilds a broken environment'
# an update over a venv made on another Python must rebuild it, not skip on the stamp
$cfg = "$Root\venv\pyvenv.cfg"
(Get-Content -LiteralPath $cfg) -replace '^home = .*', 'home = C:\nowhere\cpython-3.12-windows-x86_64-none' | Set-Content -LiteralPath $cfg
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour auto -Log $log | Out-Null
$ok = & "$Root\venv\Scripts\python.exe" -c "import diffusers, webview; print('ok')"
Check ($LASTEXITCODE -eq 0 -and $ok -eq 'ok' -and (Select-String -LiteralPath $cfg -SimpleMatch "home = $Root\python" -Quiet)) 'a venv on another Python is rebuilt'
$m = New-Object Threading.Mutex($false, 'Local\aiVibeStudioSingleInstance')
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\setup.ps1" -Root $Root -Flavour auto -Force -Log $log | Out-Null
Check ($LASTEXITCODE -eq 2) 'Repair refuses while the app is open'
$m.Dispose()
Write-Output 'ALL OK'
