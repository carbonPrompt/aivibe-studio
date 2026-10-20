"""The installer script ships what the app needs and nothing tied to one PC."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ISS = (ROOT / "installer" / "aivibe.iss").read_text(encoding="utf-8")
SETUP = (ROOT / "installer" / "setup.ps1").read_text(encoding="utf-8")


def test_iss_has_mutex_and_lowest_privileges():
    assert "AppMutex=Local\\aiVibeStudioSingleInstance" in ISS
    assert "PrivilegesRequired=lowest" in ISS
    assert "DefaultDirName={localappdata}\\Programs\\aiVibe Studio" in ISS
    assert "AppPublisher=carbonPrompt" in ISS


def test_iss_no_absolute_paths():
    assert not re.search(r"[A-Za-z]:\\|/home/|\\Users\\", ISS)


def test_iss_ships_all_runtime_folders():
    for src in ('..\\app.pyw', '..\\studio\\*', '..\\ui\\*', '..\\examples\\*', 'bin\\uv.exe',
                'requirements.lock.txt', 'setup.ps1', '..\\LICENSE', '..\\NOTICE'):
        assert f'Source: "{src}"' in ISS, src


def test_iss_creates_no_background_items():
    for word in ("[Registry]", "{userstartup}", "{commonstartup}", "schtasks", "Register-ScheduledTask", "sc.exe"):
        assert word not in ISS, word
    assert "Register-ScheduledTask" not in SETUP and "schtasks" not in SETUP


def test_setup_pins_torch_and_flavours():
    assert "torch==2.14.1+$Flavour" in SETUP and "torchvision==0.29.1+$Flavour" in SETUP
    assert "https://download.pytorch.org/whl/$Flavour" in SETUP
    assert re.search(r"ValidateSet\('cu126', ?'cu130', ?'cpu', ?'auto'\)", SETUP)


def test_lock_file_has_no_torch():
    lock = (ROOT / "installer" / "requirements.lock.txt").read_text()
    assert not re.search(r"(?m)^torch(vision)?==", lock)


def test_repair_really_reinstalls_and_refuses_while_the_app_runs():
    assert re.search(r'Name: "\{group\}\\Repair aiVibe Studio";.*-Flavour auto -Force -Pause', ISS)
    assert "[switch]$Force" in SETUP and "--reinstall" in SETUP
    assert "TryOpenExisting('Local\\aiVibeStudioSingleInstance'" in SETUP
    assert "Read-Host" in SETUP          # the Repair window says done/failed and waits


def test_setup_brings_its_own_python_without_uv_links():
    # uv-managed Python makes a junction that Windows refuses for a standard user under the installer
    # (os error 448), so setup unpacks a pinned python-build-standalone build itself
    assert "$env:UV_PYTHON_DOWNLOADS = 'never'" in SETUP and "Get-FileHash" in SETUP and "python.version" in SETUP
    assert "'--python', $pyexe" in SETUP or "'--python', $pyExe" in SETUP
    assert 'Source: "python.version"' in ISS
    url, sha = (ROOT / "installer" / "python.version").read_text().split()
    assert url.startswith("https://github.com/astral-sh/python-build-standalone/releases/download/")
    assert url.endswith("-x86_64-pc-windows-msvc-install_only.tar.gz") and re.fullmatch(r"[0-9a-f]{64}", sha)


def test_setup_rebuilds_a_venv_made_on_another_python():
    # an earlier build made the venv on a uv-managed Python behind a junction; after the update its
    # launcher could not start ("uv trampoline failed to spawn Python child process"), and the matching
    # stamp made setup skip. The stamp now covers the Python pin and the venv's home is checked.
    assert "python.version" in SETUP.split("$want")[1].split("\n")[0]
    assert "pyvenv.cfg" in SETUP and "Remove-Item -LiteralPath $venv -Recurse -Force" in SETUP
