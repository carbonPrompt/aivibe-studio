import json
import os
import time
from studio.gpu import BusyMarker, NullMarker, make_marker, other_job_running

ALIVE = {os.getpid(), 4242}
alive = lambda pid: pid in ALIVE


def test_acquire_and_release(tmp_path):
    m = BusyMarker(tmp_path / "studio.busy", pid_alive=alive)
    m.acquire()
    d = json.loads((tmp_path / "studio.busy").read_text())
    assert d["pid"] == os.getpid() and abs(d["heartbeat"] - time.time()) < 5
    m.release()
    d = json.loads((tmp_path / "studio.busy").read_text())
    assert d["state"] == "idle" and d["pid"] == os.getpid()   # still holds a CUDA context, but no model
    m.clear()
    assert not (tmp_path / "studio.busy").exists()
    m.clear()  # twice is fine


def test_heartbeat_updates(tmp_path):
    m = BusyMarker(tmp_path / "b", pid_alive=alive)
    m.acquire()
    (tmp_path / "b").write_text(json.dumps({"pid": os.getpid(), "heartbeat": 1.0}))
    m.heartbeat()
    assert json.loads((tmp_path / "b").read_text())["heartbeat"] > 1.0


def _other(tmp_path, pid, hb):
    (tmp_path / "b").write_text(json.dumps({"pid": pid, "heartbeat": hb}))
    return BusyMarker(tmp_path / "b", pid_alive=alive)


def test_own_marker_is_not_other(tmp_path):
    assert not _other(tmp_path, os.getpid(), time.time()).is_held_by_other()


def test_live_fresh_other(tmp_path):
    assert _other(tmp_path, 4242, time.time()).is_held_by_other()


def test_stale_heartbeat_ignored(tmp_path):
    assert not _other(tmp_path, 4242, time.time() - 121).is_held_by_other()


def test_dead_pid_ignored(tmp_path):
    assert not _other(tmp_path, 999999, time.time()).is_held_by_other()


def test_garbage_marker_ignored(tmp_path):
    (tmp_path / "b").write_text("nope")
    assert not BusyMarker(tmp_path / "b", pid_alive=alive).is_held_by_other()


def test_other_job_running(tmp_path, monkeypatch):
    import studio.gpu as g
    monkeypatch.setattr(g, "_is_job_process", alive)
    monkeypatch.setattr(g, "_started", lambda pid: 0.0)
    assert not other_job_running(tmp_path / "missing")
    (tmp_path / "j1").mkdir(); (tmp_path / "j1" / "pid").write_text("999999\n")
    assert not other_job_running(tmp_path)
    (tmp_path / "j2").mkdir(); (tmp_path / "j2" / "pid").write_text("4242\r\n")
    assert other_job_running(tmp_path)


def test_job_process_must_be_powershell(monkeypatch):
    import studio.gpu as g
    class P:
        def __init__(self, pid): self.pid = pid
        def name(self): return {1: "powershell.exe", 2: "chrome.exe"}[self.pid]
    monkeypatch.setattr(g.psutil, "Process", P)
    assert g._is_job_process(1) and not g._is_job_process(2)


def test_heartbeat_write_failure_is_ignored(tmp_path, monkeypatch):
    m = BusyMarker(tmp_path / "b", pid_alive=alive)
    m.acquire()
    import studio.gpu as g
    def boom(a, b): raise PermissionError("sharing violation")
    monkeypatch.setattr(g.os, "replace", boom)
    m.heartbeat()   # must not raise


def test_idle_marker_is_not_busy(tmp_path):
    (tmp_path / "b").write_text(json.dumps({"pid": 4242, "heartbeat": time.time(), "state": "idle"}))
    assert not BusyMarker(tmp_path / "b", pid_alive=alive).is_held_by_other()


def test_reused_pid_is_not_another_job(tmp_path, monkeypatch):
    # live finding 2026-10-18: an old job's pid file named a PID Windows had since given to an unrelated
    # PowerShell (our own ssh session), so the app waited 10 minutes for a job that had ended long ago
    import studio.gpu as g
    (tmp_path / "j").mkdir(); pf = tmp_path / "j" / "pid"; pf.write_text("4242")
    os.utime(pf, (1000, 1000))
    monkeypatch.setattr(g, "_is_job_process", lambda pid: True)
    monkeypatch.setattr(g, "_started", lambda pid: 5000.0)      # process born long after the pid file
    assert not other_job_running(tmp_path)
    monkeypatch.setattr(g, "_started", lambda pid: 998.0)       # born just before it wrote the file
    assert other_job_running(tmp_path)


def test_null_marker_noops(tmp_path):
    m = NullMarker()
    m.acquire(); m.heartbeat(); m.release(); m.clear()
    assert list(tmp_path.iterdir()) == []


def test_make_marker_blank_is_null(tmp_path):
    assert isinstance(make_marker(""), NullMarker)
    m = make_marker(str(tmp_path / "busy.json"))
    assert isinstance(m, BusyMarker) and m.path == tmp_path / "busy.json"
