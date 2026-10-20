import threading
import time
from pathlib import Path

import pytest

from studio.subjects import SubjectStore
from studio.trainer import build_command, parse_progress, steps_for, Trainer, TrainError


@pytest.fixture
def store(tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    s = st.create("Alex", "person", "man")
    for i in range(20):
        (st.root / s.id / "photos" / f"p{i}.png").write_bytes(b"x")
    s.photo_count = 20; st.update(s)
    return st


def test_steps_for():
    assert steps_for(10) == 800 and steps_for(30) == 1200 and steps_for(60) == 1600


def test_build_command(store, tmp_path):
    s = store.get("alex")
    cmd = build_command(s, store, tmp_path / "base", "python.exe", 1200, variant="fp16")
    j = " ".join(cmd)
    assert cmd[0] == "python.exe" and cmd[1].endswith("train_dreambooth_lora_sdxl.py")
    assert f"--instance_prompt photo of {s.token} man" in j and "--max_train_steps 1200" in j
    assert "--variant fp16" in j and "--use_8bit_adam" in j and "--report_to" not in j
    assert str(store.root / "alex" / "photos") in cmd and str(store.root / "alex" / "training") in cmd


def test_parse_progress():
    assert parse_progress("Steps:  25%|##4       | 148/600 [06:56<20:59,  2.79s/it, loss=0.006]") == (148, 600)
    assert parse_progress("loading weights") is None


class FakeProc:
    def __init__(self, cmd, out_dir, lines, rc, slow=0.0):
        self.cmd, self.out_dir, self.lines, self.rc, self.slow = cmd, out_dir, lines, rc, slow
        self.killed = False
        self.pid = 4242
        r, w = __import__("os").pipe()
        import os
        def feed():
            for l in lines:
                if self.killed:
                    break
                os.write(w, (l + "\r").encode())
                time.sleep(slow)
            if rc == 0 and not self.killed:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "pytorch_lora_weights.safetensors").write_bytes(b"NEW")
            os.close(w)
        self.stdout = os.fdopen(r, "rb", buffering=0)
        self._t = threading.Thread(target=feed, daemon=True); self._t.start()
    def wait(self, timeout=None):
        self._t.join(); return -9 if self.killed else self.rc
    def poll(self):
        return None if self._t.is_alive() else (-9 if self.killed else self.rc)


def make_trainer(store, tmp_path, rc=0, slow=0.0, lines=None):
    procs = []
    lines = lines or [f"Steps: {i}/4 [00:01<00:01]" for i in range(1, 5)]
    def run(cmd, **kw):
        p = FakeProc(cmd, store.root / "alex" / "training", lines, rc, slow); procs.append(p); return p
    def kill(p):
        p.killed = True
    return Trainer(store, python_exe="python.exe", base_resolver=lambda: tmp_path / "base", run=run, kill_tree=kill), procs


def test_success_installs_lora(store, tmp_path):
    t, _ = make_trainer(store, tmp_path)
    seen = []
    t.train("alex", threading.Event(), lambda s, n, eta: seen.append((s, n)))
    s = store.get("alex")
    assert store.lora_path("alex").read_bytes() == b"NEW" and s.trained and s.trained_at and s.steps == 800
    assert seen[-1] == (4, 4) and not (store.root / "alex" / "training").exists()


def test_failure_keeps_old_lora(store, tmp_path):
    store.lora_path("alex").write_bytes(b"OLD")
    t, _ = make_trainer(store, tmp_path, rc=1, lines=["Traceback: CUDA out of memory"])
    with pytest.raises(TrainError, match="out of memory"):
        t.train("alex", threading.Event(), lambda *a: None)
    assert store.lora_path("alex").read_bytes() == b"OLD" and not (store.root / "alex" / "training").exists()


def test_cancel_kills_tree_and_leaves_untrained(store, tmp_path):
    t, procs = make_trainer(store, tmp_path, slow=0.2, lines=[f"Steps: {i}/50 [x]" for i in range(50)])
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(TrainError, match="cancel"):
        t.train("alex", cancel, lambda *a: None)
    assert procs[0].killed and not store.get("alex").trained and not (store.root / "alex" / "training").exists()


def test_too_few_photos(store, tmp_path):
    s = store.create("Few", "thing", "object")
    t, _ = make_trainer(store, tmp_path)
    with pytest.raises(TrainError, match="photos"):
        t.train(s.id, threading.Event(), lambda *a: None)


def test_vendored_script_runs_without_a_tracker():
    # live finding: accelerate rejects log_with="none"; the script must pass None when no tracker is wanted
    from studio.trainer import SCRIPT
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'log_with=None if args.report_to == "none" else args.report_to' in src
    assert "init_trackers" not in src or 'if accelerator.is_main_process and args.report_to != "none"' in src


def test_train_resolves_base_at_train_time(store):
    t = Trainer(store, python_exe="py", base_resolver=lambda: None, run=lambda *a, **k: None)
    with pytest.raises(TrainError, match="RealVisXL"):
        t.train("alex", threading.Event(), lambda *a: None)
