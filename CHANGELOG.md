# aiVibe Studio 1.1.0

Video, and a Batch Maker.

- **Video tab.** Make short clips from a prompt, or animate a still picture, with 5 open video models: LTX-Video 2B (text or picture), Wan 2.1 1.3B (text), CogVideoX 2B (text) and Stable Video Diffusion XT and XT 1.1 (picture). Each downloads only when you pick it. Choose size, frames, frames per second, steps, guidance and seed; the app snaps them to what each model accepts and shows the clip length.
- **Animate this.** A button under any picture, and on each gallery thumbnail, sends it to the Video tab as the starting frame.
- Clips are saved as H.264 `.mp4` with a recipe beside them, show in the Recent strip with an MP4 badge and play in the app.
- **Batch Maker.** A full editor for batch files: paste prompts one per line, set shared defaults (model, size and aspect, count, steps, guidance, seed, upscale, negative prompt, subjects with likeness sliders, face fix, reference image), override any of them per item, reorder, duplicate or delete items, or edit the raw JSON. Open existing `.txt` and `.json` batches, Save (Ctrl+S), Check batch, or Save & Run. **+ Batch** next to Generate adds the current prompt and settings, and the assistant's review screen has **Open in Batch Maker**.
- A batch that names a video model now says so instead of failing mid-run; videos are made from the Video tab.
- Setup now installs ffmpeg (from the pinned imageio-ffmpeg package) to encode videos. Updating from 1.0.0 adds it to your existing Python environment.

Video models are big (4.5 to 29 GB), need a card with 12 GB of video memory, and run partly from system RAM, so a clip takes minutes (about 6 to 19 depending on the model). LTX-Video and Stable Video Diffusion have conditions for commercial use; see the README.

Install or update: download `aiVibeStudio-Setup-1.1.0.exe` below and run it, or press **Updates** in the app. Your pictures, subjects, settings and models are kept.

# aiVibe Studio 1.0.0

First public release. A local AI image studio for Windows PCs with an NVIDIA graphics card.

- Make pictures from a prompt with 12 open models, downloaded only when you pick them, with a progress window. Generate continues by itself when the download finishes.
- Batches from a `.txt` or `.json` file, with a recipe saved beside every picture.
- Upscale to 2x, 4x, 1080p, 4K or an exact size (Real-ESRGAN).
- Train your own subjects (people, pets, a car) and use up to three in one picture.
- Optional assistant: Claude, Gemini, OpenAI or Grok writes a batch with your own API key; nothing runs until you press Go.
- Hardware check: the app tells you which models fit your card.
- Updates: the app checks this project's GitHub releases and installs a new version only when you say so, after checking its checksum.

Install: download `aiVibeStudio-Setup-1.0.0.exe` below and run it. No admin rights needed. Windows SmartScreen may warn because the installer is not code-signed: More info, Run anyway. See the README for requirements, models and licenses.

Made in collaboration with AI agents (Claude and Gemini) and a human.
