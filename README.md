# Style Transfer Agent (GUI)

This project ships a PyQt-based assistant for applying “reference looks” to your photos. The app extracts a colour/CLIP fingerprint from reference images, renders new variants with a Filmulator-inspired engine, and lets you steer the edits with sliders, typed feedback, or optional voice commands. This README focuses solely on the GUI workflow—no CLI steps required.

## Features

- **Guided onboarding**: on launch you are prompted to add input photos and (optionally) reference images, so you never touch the filesystem manually.
- **Reference-aware styling**: OpenCLIP (ViT-B/32) + colour statistics drive the fingerprint that regrades each input.
- **Fine-grained controls**: sliders for strength, saturation, exposure, shadows/highlights, clarity, white balance, grain, etc., all on a unified `[-100, 100]` scale.
- **Natural-language feedback**: type instructions such as “more grain and warmer whites” and a local LLM (via Ollama) translates them into parameter tweaks.
- **Voice mode** *(optional)*: keep the GUI hands-free by dictating adjustments through Faster-Whisper, `sounddevice`, and `webrtcvad`.

## Prerequisites

- Python **3.10+**
- macOS / Linux with an OpenCL-capable PyTorch build (CPU works for small batches)
- Darktable is **not** required; all rendering happens via the bundled Filmulator-inspired engine

### Core dependencies

Create a virtual environment and install the required packages:

```bash
cd style-transfer-agent-iteration
python -m venv venv
source venv/bin/activate            # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` covers PyTorch, torchvision, OpenCLIP, Pillow, NumPy, PyQt6, scikit-learn, and the other runtime libraries the GUI needs.

### Optional extras

Install these only if you want the corresponding features:

| Feature | Extra packages | Notes |
| --- | --- | --- |
| Voice-driven feedback | `pip install faster-whisper sounddevice webrtcvad` | Faster-Whisper downloads the Whisper model at first run (~1.5 GB for `small`). |
| Local LLM interpretation | [Ollama](https://ollama.com/) with `llama3.1:8b` or similar | `brew install ollama` (macOS) or follow the official instructions, then run `ollama pull llama3.1:8b`. Keep the Ollama service running before launching the GUI. |

If you skip these installs, the GUI will simply hide/disable the related features (voice toggle, LLM fallback).

## Running the GUI

1. **Activate the environment** (see above).
2. Launch the application:
   ```bash
   python scripts/run_qt.py
   ```
3. When prompted:
   - Click **Add Images** and select the photos you’d like to stylize.
   - (Optional) Import one or more reference images when asked “Would you like to import reference image(s) to style from?”. Use “Skip for now” if you just want to experiment.
4. After onboarding, the main window shows:
   - A list of your inputs (left column).
   - Original vs. styled previews.
   - Feedback box + **Apply Feedback** button.
   - Slider drawer (toggle with the ▶ button) for direct numeric tweaks.
   - Optional **Voice Mode** toggle.

### Workflow tips

- **Restyle everything**: click “Re-style All” after adding new references.
- **Feedback loop**: type natural-language requests (e.g., “add warmth and boost clarity”) and press **Apply Feedback**. With Ollama running, the model translates the text into specific parameter updates; otherwise keyword rules handle the common phrases.
- **Voice commands**: enable **Voice Mode**, wait for the status label (“Listening…”), then speak commands such as “cool the white balance and add grain.” Transcribed text appears automatically in the feedback box before being applied.
- **Sliders**: each slider maps `-100 → +100` to the full adjustment range. Strength zero = baseline fingerprint; -100 removes the effect; +100 amplifies it beyond the default.
- **Saving results**: the app writes rendered images to `outputs/styled/` (mirroring the filenames of the inputs). You can open that folder directly from your file browser at any time.

## Directory layout

```
style-transfer-agent-iteration/
├── data/
│   ├── inputs/           # Populated automatically when you add images
│   └── references/       # Filled when you import reference looks
├── outputs/
│   └── styled/           # Rendered previews saved here
├── scripts/
│   └── run_qt.py         # Entry point for the GUI
├── src/                  # Application code (GUI, agent logic, Filmulator engine, LLM + voice adapters)
└── tests/                # Pytest suite (unit + integration)
```

## Testing

Run the tests from the project root (while the virtual environment is active):

```bash
python -m pytest
```

The suite covers the Filmulator engine, parameter mapping, GUI helpers (where practical), and the LLM/voice adapters. Add new tests alongside any substantial code changes to keep the feedback loop fast.

## Open Source Credits

### Dependencies and Acknowledgments

**Machine Learning & Computer Vision**
- **[PyTorch](https://github.com/pytorch/pytorch)** (BSD-3-Clause) and **[torchvision](https://github.com/pytorch/vision)** (BSD-3-Clause): Deep learning framework powering CLIP feature extraction and tensor transformations.
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** (MIT): Provides ViT-B/32 model weights and preprocessing pipeline for the semantic fingerprint extractor.
- **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** (BSD-3-Clause): Machine learning utilities supporting normalization and experimental feature engineering.

**Image Processing**
- **[NumPy](https://github.com/numpy/numpy)** (BSD-3-Clause): Array operations and numerical computations for pixel-level manipulation.
- **[Pillow](https://github.com/python-pillow/Pillow)** (HPND): Image file I/O, format conversions, and basic transformations.
- The tone-mapping pipeline draws inspiration from the open-source **[Filmulator](https://github.com/CarVac/filmulator-gui)** project (GPLv3). This implementation is a clean-room reimplementation employing similar highlight/shadow recovery techniques and grain simulation.

**User Interface**
- **[PyQt6](https://riverbankcomputing.com/software/pyqt/)** (GPL/Commercial): Cross-platform GUI framework for the desktop application.

**Audio Processing & Natural Language**
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** (MIT): Real-time speech-to-text transcription engine optimized for local inference.
- **[sounddevice](https://github.com/spatialaudio/python-sounddevice)** (MIT): Low-latency audio capture from system microphones.
- **[webrtcvad](https://github.com/wiseman/py-webrtcvad)** (MIT): Voice activity detection for intelligent audio segmentation.
- **[Ollama](https://github.com/ollama/ollama)** (MIT): LLM serving infrastructure enabling local Llama-family models for natural language feedback parsing.


## Troubleshooting

- **No images listed after onboarding**: the app only accepts `.jpg`, `.jpeg`, `.png`, or `.bmp`. Re-run “Load Inputs” from the main toolbar if you skipped the first dialog.
- **LLM feedback ignored**: ensure the Ollama daemon is running (`ollama serve` or the default background service) and that the requested model (default `llama3.1:8b`) is already pulled.
- **Voice mode errors**: confirm your microphone permissions allow access, then reinstall `sounddevice` with the appropriate backend (e.g., `pip install sounddevice==0.4.6`). The GUI shows toast-style warnings when audio capture fails.

Code generated utilising the SOTA coding-oritentated model GPT-5 Codex.
