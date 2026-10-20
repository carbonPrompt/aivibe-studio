# aiVibe Studio's Python environment: Python 3.12, PyTorch and the pinned libraries, built next to the app.
# Run by the installer after copying the files (hidden), and by "Repair aiVibe Studio" in the Start menu
# (-Force -Pause: reinstall everything, show progress, wait for a key at the end).
# Idempotent without -Force: when nothing changed since the last good run it finishes in seconds.
param(
  [Parameter(Mandatory = $true)][string]$Root,
  [ValidateSet('cu126', 'cu130', 'cpu', 'auto')][string]$Flavour = 'auto',
  [string]$Log = '',
  [switch]$Force,
  [switch]$Pause
)
# PS 5.1 turns native stderr into errors under 'Stop'; exit codes are checked by hand instead.
$ErrorActionPreference = 'Continue'
$Root = $Root.TrimEnd('\')
if (-not $Log) { $Log = Join-Path $env:LOCALAPPDATA 'aiVibe Studio\data\logs\setup.log' }
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null

function Say([string]$text) {
  $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $text
  Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
  Write-Host $line
}

function Finish([int]$code) {
  if ($Pause) {
    if ($code) { Write-Host "`nRepair did not finish. Details: $Log" } else { Write-Host "`nRepair finished. You can start aiVibe Studio again." }
    Read-Host 'Press Enter to close this window' | Out-Null
  }
  exit $code
}

function Run([string]$what, [string[]]$argv) {
  Say "$what"
  & $uv @argv 2>&1 | ForEach-Object { Add-Content -LiteralPath $Log -Value "    $_" -Encoding UTF8 }
  if ($LASTEXITCODE) { Say "FAILED: $what (exit $LASTEXITCODE)"; Finish $LASTEXITCODE }
}

# never rebuild the environment under a running app
$running = $null
if ([Threading.Mutex]::TryOpenExisting('Local\aiVibeStudioSingleInstance', [ref]$running)) {
  $running.Dispose()
  Say 'aiVibe Studio is open. Close it, then run Repair again.'
  Finish 2
}

$flavourFile = Join-Path $Root 'torch.txt'
if ($Flavour -eq 'auto') {
  $Flavour = if (Test-Path -LiteralPath $flavourFile) { (Get-Content -LiteralPath $flavourFile -Raw).Trim() } else { 'cu126' }
}
Set-Content -LiteralPath $flavourFile -Value $Flavour -Encoding ASCII

$uv = Join-Path $Root 'uv.exe'
$venv = Join-Path $Root 'venv'
$py = Join-Path $venv 'Scripts\python.exe'
$lock = Join-Path $Root 'requirements.lock.txt'
$env:UV_CACHE_DIR = Join-Path $Root 'uvcache'
$env:UV_NO_PROGRESS = '1'
# uv never fetches or links Python itself: its managed installs make a folder junction that Windows refuses
# for a standard user under the installer (os error 448). Setup unpacks a pinned standalone build instead.
$env:UV_PYTHON_DOWNLOADS = 'never'
$pyHome = Join-Path $Root 'python'
$pyExe = Join-Path $pyHome 'python.exe'

$stamp = Join-Path $venv '.lockhash'
$sha = [Security.Cryptography.SHA256]::Create()
$want = [BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($lock) + [IO.File]::ReadAllBytes((Join-Path $Root 'python.version')))).Replace('-', '') + "-$Flavour"
# a venv made on any other Python (an earlier build used a uv-managed one behind a junction) cannot start: rebuild it
$cfg = Join-Path $venv 'pyvenv.cfg'
if ((Test-Path -LiteralPath $venv) -and -not ((Test-Path -LiteralPath $cfg) -and
    (Select-String -LiteralPath $cfg -SimpleMatch "home = $pyHome" -Quiet) -and (Test-Path -LiteralPath $pyExe))) {
  Say 'the Python environment was made on another Python; rebuilding it'
  Remove-Item -LiteralPath $venv -Recurse -Force
  Get-ChildItem -LiteralPath $pyHome -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'cpython-*' -or $_.Name -eq '.temp' } |
    ForEach-Object { if ($_.LinkType) { $_.Delete() } else { Remove-Item -LiteralPath $_.FullName -Recurse -Force } }
}
if (-not $Force -and (Test-Path -LiteralPath $stamp) -and (Test-Path -LiteralPath (Join-Path $venv 'Scripts\pythonw.exe')) -and
    ((Get-Content -LiteralPath $stamp -Raw).Trim() -eq $want)) {
  Say "environment already up to date ($Flavour)"
  Finish 0
}

Say "setting up in $Root, PyTorch build $Flavour$(if ($Force) { ', reinstalling everything' })"
$again = if ($Force) { @('--reinstall') } else { @() }
if (-not (Test-Path -LiteralPath $pyExe)) {
  $pin = (Get-Content -LiteralPath (Join-Path $Root 'python.version') -Raw).Trim() -split '\s+'
  $tgz = Join-Path $Root 'python.tar.gz'
  Say 'Python 3.12 (standalone build)'
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $pin[0] -OutFile $tgz -UseBasicParsing
  } catch { Say "FAILED: downloading Python ($($_.Exception.Message))"; Finish 1 }
  if ((Get-FileHash -LiteralPath $tgz -Algorithm SHA256).Hash -ne $pin[1].ToUpper()) {
    Remove-Item -LiteralPath $tgz -Force; Say 'FAILED: the Python download did not match its checksum'; Finish 1
  }
  Remove-Item -LiteralPath $pyHome -Recurse -Force -ErrorAction SilentlyContinue
  & "$env:SystemRoot\System32\tar.exe" -xzf $tgz -C $Root        # unpacks to <Root>\python
  Remove-Item -LiteralPath $tgz -Force
  if (-not (Test-Path -LiteralPath $pyExe)) { Say 'FAILED: unpacking Python'; Finish 1 }
}
if (-not (Test-Path -LiteralPath $py)) {
  Run 'a fresh environment' @('venv', '--python', $pyExe, $venv)
}
Run "PyTorch 2.14.1 ($Flavour), the largest download" (@('pip', 'install', '--python', $py, '--no-deps') + $again + @(
  '--index-url', "https://download.pytorch.org/whl/$Flavour", "torch==2.14.1+$Flavour", "torchvision==0.29.1+$Flavour"))
Run 'the pinned libraries' (@('pip', 'install', '--python', $py, '--no-deps') + $again + @('-r', $lock))
Run 'the window (pywebview)' (@('pip', 'install', '--python', $py) + $again + @('--exclude-newer', '2026-10-01', 'pywebview>=5,<7'))

Say 'checking the environment'
$check = & $py -c "import torch, diffusers, webview; print('cuda', torch.cuda.is_available(), torch.__version__)" 2>&1
$check | ForEach-Object { Add-Content -LiteralPath $Log -Value "    $_" -Encoding UTF8 }
if ($LASTEXITCODE) { Say 'FAILED: the libraries do not import'; Finish 1 }
if ($Flavour -ne 'cpu' -and -not ($check -match 'cuda True')) {
  Say 'WARNING: PyTorch does not see the NVIDIA card; pictures will be made on the CPU'
}
Set-Content -LiteralPath $stamp -Value $want -Encoding ASCII
Say 'done'
Finish 0
