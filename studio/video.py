"""Video helpers: frame conversion, MP4 encoding (ffmpeg H.264 with OpenCV fallback), thumbnailing and parameter snapping."""
import io
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("studio.video")
NO_WINDOW = 0x08000000
VIDEO_EXT = (".mp4", ".webm")
VIDEO_FAMILIES = {"ltx", "wan", "cogvideo", "svd"}
VIDEO_I2V_FAMILIES = {"ltx", "svd"}
VIDEO_T2V_FAMILIES = {"ltx", "wan", "cogvideo"}


@dataclass
class VideoRequest:
    model_id: str
    prompt: str
    negative: str
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    guidance: float
    seed: int | None
    image: str | None = None        # optional starting frame path for Image-to-Video


def snap_video_params(family: str, width: int, height: int, frames: int, fps: int) -> tuple[int, int, int, int]:
    """Snap resolution and frame count to the multiples each video architecture requires."""
    fps = max(4, min(60, int(fps or 16)))
    if family == "ltx":
        w = max(256, min(1280, round(width / 32) * 32))
        h = max(256, min(1280, round(height / 32) * 32))
        k = max(1, min(20, round((max(9, frames) - 1) / 8)))
        f = k * 8 + 1
    elif family == "wan":
        w = max(256, min(1280, round(width / 16) * 16))
        h = max(256, min(1280, round(height / 16) * 16))
        k = max(2, min(30, round((max(9, frames) - 1) / 4)))
        f = k * 4 + 1
    elif family == "cogvideo":
        w = max(256, min(1280, round(width / 16) * 16))
        h = max(256, min(1280, round(height / 16) * 16))
        k = max(1, min(10, round((max(9, frames) - 1) / 8)))
        f = k * 8 + 1
    elif family == "svd":
        w = max(256, min(1280, round(width / 64) * 64))
        h = max(256, min(1280, round(height / 64) * 64))
        f = max(14, min(25, int(frames or 25)))
    else:
        w = max(256, min(1280, round(width / 16) * 16))
        h = max(256, min(1280, round(height / 16) * 16))
        f = max(8, min(161, int(frames or 49)))
    return w, h, f, fps


def to_pil_frames(raw) -> list[Image.Image]:
    """Turn a diffusers video output (list of PIL images, nested list, or numpy array) into a flat list of RGB PIL images."""
    if isinstance(raw, np.ndarray) and raw.ndim == 5 and raw.shape[0] == 1:
        raw = raw[0]
    elif isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple, np.ndarray)):
        raw = raw[0]
    out = []
    for frame in raw:
        if isinstance(frame, Image.Image):
            out.append(frame.convert("RGB"))
        else:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0 if arr.max() <= 1.05 else arr, 0, 255).astype(np.uint8)
            out.append(Image.fromarray(arr).convert("RGB"))
    return out


def find_ffmpeg() -> str | None:
    """Locate ffmpeg: beside the active Python, the imageio-ffmpeg build the installer pins, or PATH."""
    exe_dir = Path(sys.executable).resolve().parent
    for name in ("ffmpeg.exe", "ffmpeg"):
        cand = exe_dir / name
        if cand.is_file():
            return str(cand)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def save_video(frames: list[Image.Image], path: Path, fps: int = 16) -> None:
    """Encode a list of RGB PIL frames into an .mp4 file using H.264 ffmpeg when available, or OpenCV VideoWriter."""
    if not frames:
        raise ValueError("no video frames to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    # Ensure even dimensions for YUV420p video encoders
    w2, h2 = w - (w % 2), h - (h % 2)
    norm = [im.convert("RGB") if im.size == (w2, h2) else im.convert("RGB").resize((w2, h2), Image.LANCZOS) for im in frames]

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w2}x{h2}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "fast", "-movflags", "+faststart", str(path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=NO_WINDOW if os.name == "nt" else 0)
            for im in norm:
                proc.stdin.write(im.tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace").strip()
            if proc.wait(timeout=120) == 0 and path.exists() and path.stat().st_size > 0:
                return
            log.warning("ffmpeg encoding failed (%s), falling back to OpenCV", err)
        except Exception as e:
            log.warning("ffmpeg failed (%s), falling back to OpenCV", e)

    import cv2
    for fourcc_name in ("vp09", "mp4v"):
        path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), (w2, h2))
        if writer.isOpened():
            try:
                for im in norm:
                    arr = np.asarray(im, dtype=np.uint8)[:, :, ::-1]   # RGB -> BGR
                    writer.write(arr)
            finally:
                writer.release()
            if path.exists() and path.stat().st_size > 0:
                return
    raise RuntimeError("could not encode video to MP4")


def video_first_frame(path: Path) -> Image.Image:
    """Read the first frame of a saved video as a PIL Image for gallery thumbnails."""
    # Gallery files may come from anywhere: force the container demuxer and allow only local file reads, so a
    # playlist renamed to .mp4 cannot make ffmpeg (or OpenCV's ffmpeg) fetch URLs or read other local files.
    demux = "matroska" if Path(path).suffix.lower() == ".webm" else "mov"
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        try:
            r = subprocess.run([ffmpeg, "-loglevel", "error", "-protocol_whitelist", "file", "-f", demux,
                                "-i", str(path), "-frames:v", "1",
                                "-f", "image2pipe", "-vcodec", "png", "-"],
                               capture_output=True, timeout=15,
                               creationflags=NO_WINDOW if os.name == "nt" else 0)
            if r.returncode == 0 and r.stdout:
                with Image.open(io.BytesIO(r.stdout)) as im:
                    return im.convert("RGB")
        except Exception:
            pass
    import cv2
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file"
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot read video frame from {path.name}")
        return Image.fromarray(frame[:, :, ::-1]).convert("RGB")
    finally:
        cap.release()
