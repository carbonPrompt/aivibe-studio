from pathlib import Path, PureWindowsPath

from studio import paths


def test_data_root_env_override(tmp_path):
    assert paths.data_root({"AIVIBE_DATA": str(tmp_path), "LOCALAPPDATA": "C:\\x"}) == Path(tmp_path)


def test_data_root_localappdata_unicode():
    root = paths.data_root({"LOCALAPPDATA": "C:\\Users\\José Ñ\\AppData\\Local"})
    assert PureWindowsPath(str(root)) == PureWindowsPath("C:\\Users\\José Ñ\\AppData\\Local\\aiVibe Studio")


def test_default_hf_home(tmp_path):
    assert paths.default_hf_home(tmp_path) == tmp_path / "models"


def test_model_folder_setting_wins_over_a_global_hf_home(tmp_path):
    assert paths.hf_home("D:\\hf", tmp_path) == Path("D:\\hf")
    assert paths.hf_home("", tmp_path) == tmp_path / "models"
    src = (Path(__file__).parent.parent / "app.pyw").read_text(encoding="utf-8")
    assert 'os.environ["HF_HOME"] = str(hf_home(' in src and 'setdefault("HF_HOME"' not in src
