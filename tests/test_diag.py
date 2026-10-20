import inspect
import json
import logging
from datetime import datetime

import pytest
from PIL import Image

from studio import diag

NOW = datetime(2026, 10, 19, 9, 30, 0)


def test_log_tail_keeps_recent_lines_and_their_traces(tmp_path):
    (tmp_path / "studio.log.2026-10-18").write_text("2026-10-18 23:59:00 INFO x: yesterday\n", encoding="utf-8")
    (tmp_path / "studio.log").write_text(
        "2026-10-19 08:00:00 INFO x: too old\n"
        "  old trace line\n"
        "2026-10-19 09:10:00 ERROR x: boom\n"
        "Traceback (most recent call last):\n"
        "  File \"a.py\", line 1\n"
        "2026-10-19 09:29:00 INFO x: latest\n", encoding="utf-8")
    tail = diag.log_tail(tmp_path, 30, NOW)
    assert "too old" not in tail and "old trace line" not in tail and "yesterday" not in tail
    assert "boom" in tail and "Traceback" in tail and 'File "a.py"' in tail and "latest" in tail


def test_write_report_bundles_everything(tmp_path):
    logs, reports, out = tmp_path / "logs", tmp_path / "reports", tmp_path / "out"
    logs.mkdir(); out.mkdir()
    (logs / "studio.log").write_text("2026-10-19 09:25:00 INFO x: cancel pressed\n", encoding="utf-8")
    img = out / "generated_dog_2026-10-19_0925.png"
    Image.new("RGB", (8, 8)).save(img)
    img.with_suffix(".json").write_text('{"prompt": "a dog"}', encoding="utf-8")
    shot = reports / "_pending.png"
    reports.mkdir(); Image.new("RGB", (4, 4)).save(shot)
    folder = diag.write_report(reports, logs, "  cancel did nothing  ", version="0.3.1", settings={"theme": "dark"},
                               state={"running": True}, events=[{"seq": 1, "kind": "queued"}], image=img,
                               screenshot=shot, now=NOW)
    assert folder.name == "2026-10-19_093000"
    assert (folder / "note.txt").read_text(encoding="utf-8") == "cancel did nothing\n"
    assert "cancel pressed" in (folder / "log.txt").read_text(encoding="utf-8")
    assert (folder / img.name).exists() and (folder / img.with_suffix(".json").name).exists()
    assert (folder / "screen.png").exists() and not shot.exists()
    info = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert info["version"] == "0.3.1" and info["state"] == {"running": True} and info["events"][0]["kind"] == "queued"
    again = diag.write_report(reports, logs, "second", version="0.3.1", settings={}, state={}, events=[], now=NOW)
    assert again.name == "2026-10-19_093000_2"
    assert [r["note"] for r in diag.list_reports(reports)] == ["second", "cancel did nothing"]


def test_logged_api_keeps_parameter_names_and_logs_failures(caplog):
    @diag.logged_api(quiet={"poll"})
    class A:
        def go(self, path, count=1):
            return count
        def poll(self, since):
            return since
        def bad(self):
            raise RuntimeError("nope")

    a = A()
    assert inspect.getfullargspec(a.go).args == ["self", "path", "count"]     # what pywebview reads
    with caplog.at_level(logging.INFO, logger="studio.api"):
        assert a.go("x.png", count=2) == 2
        assert a.poll(5) == 5
        with pytest.raises(RuntimeError):
            a.bad()
    text = caplog.text
    assert "call go('x.png', count=2)" in text and "poll" not in text
    assert "call bad failed" in text and "RuntimeError: nope" in text


def test_short_hides_image_data():
    assert diag.short("data:image/png;base64," + "A" * 5000) == "<data 5022 chars>"
    assert diag.short("x" * 400).endswith("...(+102)")
