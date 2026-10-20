import json

from studio import update

DL = "https://github.com/carbonPrompt/aivibe-studio/releases/download/"


def release(tag="v1.2.0", assets=None, body="What's new"):
    assets = assets if assets is not None else [
        {"name": "source.zip", "browser_download_url": "https://x/source.zip", "size": 10},
        {"name": f"aiVibeStudio-Setup-{tag.lstrip('v')}.exe", "size": 4242, "digest": "sha256:" + "ab" * 32,
         "browser_download_url": f"{DL}{tag}/aiVibeStudio-Setup-{tag.lstrip('v')}.exe"}]
    return json.dumps({"tag_name": tag, "body": body, "assets": assets}).encode()


def test_parse_version_v_prefix():
    assert update.parse_version("v1.2.3") == (1, 2, 3) and update.parse_version("0.9.0") == (0, 9, 0)


def test_newer_true_false_equal():
    info = {"version": "1.2.0"}
    assert update.newer("1.1.9", info) is True
    assert update.newer("1.2.0", info) is False
    assert update.newer("1.10.0", info) is False
    assert update.newer("1.0.0", None) is False


def test_latest_picks_matching_asset():
    seen = {}
    def fetch(url, headers, timeout):
        seen.update(url=url, ua=headers["User-Agent"], timeout=timeout)
        return release()
    info = update.latest(fetch=fetch)
    assert info == {"version": "1.2.0", "url": f"{DL}v1.2.0/aiVibeStudio-Setup-1.2.0.exe", "notes": "What's new",
                    "size": 4242, "sha256": "ab" * 32}
    assert seen["url"] == "https://api.github.com/repos/carbonPrompt/aivibe-studio/releases/latest"
    assert seen["ua"].startswith("aiVibe-Studio/") and seen["timeout"] == 10


def test_latest_without_installer_asset_is_none():
    assert update.latest(fetch=lambda u, h, t: release(assets=[])) is None


def test_latest_offline_returns_none():
    def fetch(url, headers, timeout):
        raise OSError("no route to host")
    assert update.latest(fetch=fetch) is None


def test_latest_garbage_returns_none():
    assert update.latest(fetch=lambda u, h, t: b"<html>rate limited</html>") is None


def test_latest_refuses_installer_from_elsewhere():
    bad = [{"name": "aiVibeStudio-Setup-1.2.0.exe", "browser_download_url": "https://evil.example/aiVibeStudio-Setup-1.2.0.exe",
            "size": 1, "digest": "sha256:" + "ab" * 32}]
    assert update.latest(fetch=lambda u, h, t: release(assets=bad)) is None
    http = [dict(bad[0], browser_download_url="http://github.com/carbonPrompt/aivibe-studio/releases/download/v1.2.0/aiVibeStudio-Setup-1.2.0.exe")]
    assert update.latest(fetch=lambda u, h, t: release(assets=http)) is None


def test_latest_requires_a_sha256_digest():
    no_digest = [{"name": "aiVibeStudio-Setup-1.2.0.exe", "size": 1,
                  "browser_download_url": f"{DL}v1.2.0/aiVibeStudio-Setup-1.2.0.exe"}]
    assert update.latest(fetch=lambda u, h, t: release(assets=no_digest)) is None
