# ASL Real-Time Translator

This project is a real-time American Sign Language recognition app built around a fine-tuned I3D model from the ASL-Citizen dataset.

The app takes a live webcam feed, runs sign recognition, stabilizes the predictions so the output does not flicker constantly, and can send the final video feed into a virtual camera for apps like Google Meet or Zoom.

There is also an optional local LLM rewrite step that tries to turn recognized glosses into a more natural English sentence.

## What This Project Does

- Reads a live webcam feed
- Runs real-time ASL word recognition on a 25-gloss subset
- Smooths unstable predictions so single-frame mistakes do not immediately appear
- Displays caption-style output on the video feed
- sends the final feed into a virtual camera
- Includes a desktop control panel built with `pywebview`
- Optionally uses a local Ollama model to rewrite gloss tokens into simple English

## System Architecture

This is the full live path from the webcam to the final output feed:

```text
Webcam
  |
  v
app/capture/webcam.py
  |
  v
FramePacket
  |
  v
app/pipeline/preprocess.py
  |
  |  builds a rolling clip of frames
  v
Clip Tensor
  |
  v
app/models/i3d_word.py
  |
  |  loads checkpoint + runs I3D inference
  v
Raw probabilities
  |
  v
app/pipeline/postprocess.py
  |
  |  smoothing + thresholds + vote window + cooldown
  v
Stable label / emitted token
  |
  +-------------------------> app/output/transcript.py
  |                              |
  |                              v
  |                         running transcript
  |                              |
  |                              +-----> app/output/llm_rewrite.py (optional)
  |                                              |
  |                                              v
  |                                        English sentence
  |
  v
app/runtime/session.py
  |
  |  builds final OpenCV frame
  v
Rendered output frame
  |
  +-------------------------> local preview window (optional)
  |
  +-------------------------> app/output/virtual_cam.py (optional)
  |                              |
  |                              v
  |                         OBS / Meet / Zoom
  |
  +-------------------------> app/ui/dashboard.py
                                 |
                                 v
                          status + controls
```

### How the Live Pipeline Works

1. `webcam.py` opens the webcam and reads frames.
2. Each frame is wrapped into a `FramePacket` with a frame id and timestamp.
3. `preprocess.py` stores the recent frames in a rolling clip buffer.
4. Once enough frames are collected, `i3d_word.py` converts the clip into the shape expected by the I3D model and runs inference.
5. The model returns raw class probabilities, not a final trusted answer yet.
6. `postprocess.py` smooths the predictions over time and checks:
   - confidence
   - ambiguity between top predictions
   - repeated agreement over recent predictions
   - cooldown before registering a new token
7. If the prediction is stable enough, a token is emitted into `transcript.py`.
8. If LLM rewriting is enabled, `llm_rewrite.py` can take the transcript and try to turn it into a short English sentence.
9. `session.py` draws the final caption text onto the OpenCV frame.
10. That final frame can go to:
    - the local preview window
    - the dashboard status display
    - the virtual camera output

### Why the Pipeline Is Split This Way

I separated the system into stages because each stage solves a different problem:

- `capture` handles device input
- `preprocess` prepares data for the model
- `models` runs inference
- `postprocess` turns noisy predictions into usable output
- `output` handles transcript, sentence rewrite, and virtual camera
- `runtime` ties everything together
- `ui` gives the user control without touching code

This separation made it easier to debug the project because I could test:

- model quality separately
- live pipeline quality separately
- sentence rewriting separately
- virtual camera behavior separately

## Why I Built It This Way

Real-time sign recognition is not just a model problem. The harder part is making the output usable in a live setting.

A raw model prediction changes constantly from frame to frame, especially during:

- hand movement into position
- transitions between signs
- partial signs
- brief uncertain moments

To make the app usable, I added a post-processing layer that:

- smooths predictions over time
- checks confidence and uncertainty 
- requires repeated agreement before output is accepted
- adds cooldown logic so the same token is not emitted repeatedly

That means the project is not just "webcam to model". It is a full runtime pipeline designed for live use.

## Main Features

### 1. Real-Time ASL Recognition

The app uses an I3D-based sign recognition model fine-tuned on a top-25 ASL-Citizen subset.

### 2. Stabilized Output

The app does not trust a single frame. It uses:

- EMA smoothing
- voting over recent predictions
- confidence thresholds
- margin thresholds
- anti-repeat logic

This was one of the biggest practical improvements in the project.

### 3. Desktop Dashboard

The UI lets the user:

- start and stop the session
- clear the transcript
- change runtime settings
- enable or disable the virtual camera
- enable or disable LLM rewriting
- adjust thresholds and smoothing behavior

### 4. Virtual Camera Output

The final rendered feed can be sent into a virtual camera for use in:

- Google Meet
- Zoom
- other camera based apps

### 5. Optional Local LLM Rewrite

The app can send recognized gloss tokens to a local Ollama model to rewrite them into a short English sentence.

This feature is optional because gloss-to-English rewriting is much less reliable when the recognized tokens are noisy or incomplete.

## Project Structure

```text
ASL-RealTime-Translator/
├── app/
│   ├── capture/
│   │   └── webcam.py              # Opens the webcam, probes camera indexes when needed, and returns frames as FramePacket objects.
│   ├── core/
│   │   ├── config.py              # Loads configs/models.yaml and resolves relative paths like checkpoints/ and CSV files into full absolute paths.
│   │   ├── events.py              # Small event-bus helper kept for future event-driven architecture work. Not part of the current main loop.
│   │   └── types.py               # Shared data types such as Prediction, FramePacket, and TranscriptToken.
│   ├── models/
│   │   ├── base.py                # Common predictor interface used by model wrapper classes.
│   │   ├── i3d_word.py            # Main word-recognition wrapper. Loads the checkpoint, builds the gloss dictionary, prepares clips, and returns predictions.
│   │   ├── pytorch_i3d.py         # The actual I3D neural network architecture used by the app.
│   │   ├── TestI3d.py             # Small developer test script used to verify that the model wrapper loads and returns outputs correctly.
│   │   └── i3d_letter.py          # Placeholder file for a future letter-recognition model. Not implemented yet.
│   ├── output/
│   │   ├── llm_rewrite.py         # Optional Ollama integration. Sends transcript text to a local LLM and asks for a short English sentence.
│   │   ├── transcript.py          # Stores stable emitted tokens and builds the running transcript shown by the app.
│   │   └── virtual_cam.py         # Sends the final OpenCV frame into a virtual camera for Google Meet, Zoom, and similar apps.
│   ├── pipeline/
│   │   ├── preprocess.py          # Builds rolling clips from live frames so the model receives a short sequence instead of one frame.
│   │   ├── postprocess.py         # Stabilizes live predictions using smoothing, confidence thresholds, vote windows, cooldown logic, and anti-repeat handling.
│   │  
│   ├── runtime/
│   │   ├── session.py             # Main live pipeline. Connects webcam input, clip buffering, model inference, postprocessing, transcript, optional LLM, and virtual camera.
│   │   └── mode_manager.py        # Placeholder for future mode switching such as word mode vs letter mode. Not used right now.
│   ├── ui/
│   │   └── dashboard.py           # Desktop control panel built with pywebview. Lets the user start/stop the session, change settings, and monitor status.
│   └── main.py                    # Main entrypoint. Loads config, builds runtime settings, and starts either terminal mode or UI mode.
├── asl_subset_top25/
│   ├── selected_glosses.txt       # List of the top-25 glosses selected from the most frequent training classes.
│   ├── subset_summary.json        # Summary metadata showing how the subset was created.
│   └── splits/
│       ├── train.csv              # Training split for the top-25 subset used by the current word model.
│       ├── val.csv                # Validation split for tuning and checking performance during development.
│       └── test.csv               # Test split for final clean evaluation.
├── checkpoints/
│   └── word/
│       ├── best_v1.pt             # Main checkpoint currently used by the app.
│       └── _v174_0.741945.pt      # Older checkpoint kept from an earlier stage of training.
├── configs/
│   └── models.yaml                # Main config file. Stores model paths and runtime settings such as camera, thresholds, smoothing, virtual camera, and LLM options.
├── evaluate/
│   ├── eval_model.py              # Dataset-style parity evaluation. Used to verify that the checkpoint still performs correctly on the clean subset clips.
│   ├── eval_webcam_clips.py       # Realtime-style evaluation. Used to test how the full live pipeline behaves on video clips.
│   └── test_llm_rewrite.py        # Small script for testing local LLM rewrite behavior outside the main app.
├── outputs/                       # Stores generated evaluation outputs such as summaries and confusion matrices.
├── requirements.txt               # Python dependency list for the project.
├── videotransforms.py             # Video transform utilities used by the I3D preprocessing pipeline.
├── realtime.py                    # Earlier real-time prototype kept for comparison and experimentation.
└── README.md                      # Project overview, setup instructions, architecture notes, and usage guide.
```

## Requirements

- Python 3.11
- macOS with webcam access
- a trained model with the it's training classes 

Optional:

- `OBS` if you want virtual camera output in Meet/Zoom
- `Ollama` if you want local sentence rewriting
- `pywebview` if you want the desktop dashboard window

## Install

Create and activate a virtual environment, then install the dependencies:

```bash
cd {directory}
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Files

The app expects a checkpoint and train CSV path in `configs/models.yaml`.

Current top-25 model paths:

- checkpoint: `checkpoints/word/best_v1.pt`
- train CSV: `asl_subset_top25/splits/train.csv`

If you move these files, update `configs/models.yaml`.

## Running the App

### Terminal Mode

```bash
cd {directory}
source .venv/bin/activate
python -m app.main
```

### Desktop UI Mode

```bash
cd {directory}
source .venv/bin/activate
python -m app.main --ui webview
```

## Configuration

Main runtime settings live in:

`configs/models.yaml`

Important settings:

- `camera`: camera index, use `-1` to auto-probe
- `cam_width`, `cam_height`: webcam resolution
- `clip_len`: number of frames per inference clip 
- `infer_every`: how often inference runs (recommended is 2 setting it to 1 would cause latency and overworking your hardware)
- `threshold`: minimum confidence for a label
- `margin_threshold`: gap required between top-1 and top-2
- `vote_window`, `min_votes`: stability voting (vote_window sets how many recent predictions the app looks at,min_votes sets how many of those predictions must agree on the same label before it gets regestered.)
- `cooldown`: prevents repeated rapid emissions
- `virtual_cam_enabled`: enables virtual camera output
- `llm_enabled`: enables local sentence rewriting

## Optional: Virtual Camera Setup

To use use the realtime output you would need to use the virtual camera and have OBS installed. 

### Steps

1. Install OBS Studio
2. Open OBS at least once
3. Enable OBS Virtual Camera
4. Allow any required system permissions in macOS
5. Restart the machine if macOS asks for it

Note: if you want to view the output in OBS directly you would need to create a new "video capture source" and make sure the OBS virtual camera is selected as your device. 

After that, camera-based apps should be able to see the virtual camera.

Important:

- If virtual camera does not appear in Meet or Zoom, fully restart the browser/app after starting the virtual camera.

## Optional: Ollama Setup

The LLM rewrite feature is local, not cloud-based.

Install and run Ollama:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

The app currently uses:

- `llama3.2:3b`

Why this model:

- it is fast enough locally
- it works better for this project than the slower larger models that were tested

If you do not want sentence rewriting, keep `llm_enabled: false`.

## Evaluation

This project includes evaluation scripts in the `evaluate/` folder.

These were used to compare:

- parity-style dataset evaluation
- realtime-style webcam evaluation
- local LLM rewrite behavior

Example:

```bash
python -m evaluate.test_llm_rewrite
```

## Engineering Problems I Had To Solve

### Problem 1: Good offline accuracy did not mean good live performance

The model performed very well on clean dataset evaluation, but live webcam behavior was weaker.

Main reason:

- domain shift between dataset clips and live webcam input

What I did:

- added smoothing and stability logic
- added uncertainty handling
- avoided accepting every frame as a final answer

### Problem 2: Repeated false labels

A common live issue was the model repeating the same wrong label during transitions or uncertain moments.

What I did:

- confidence thresholding
- margin thresholding
- vote windows
- anti-repeat logic

### Problem 3: Real-time output needed to be usable in meetings

A console-only demo is not enough for an actual user.

What I did:

- built a dashboard UI
- added virtual camera support
- changed the output to a cleaner caption-style feed instead of a debug heavy overlay

### Problem 4: LLM sentence rewriting was too slow at first

Early local LLM tests were too slow and too verbose.

What I did:

- switched to a smaller local model
- tightened the prompt
- forced shorter outputs
- treated the LLM as optional instead of making the app depend on it

## Known Limitations

- The current recognition model is only trained on a 25-gloss subset
- Real-time recognition is weaker than offline evaluation
- Sentence rewriting can hallucinate if the recognized glosses are noisy
- The app is more reliable for demonstration than for production-level translation

## What I Would Improve Next

- train on a larger and more diverse gloss set
- add a `no_sign` or transition class
- improve real-time robustness with more webcam style data
- package the app into a cleaner end-user desktop installer

## Demo Notes

For the cleanest demo:

- use the UI mode
- keep `llm_enabled` optional
- keep the virtual camera enabled only if needed
- use the saved top-25 checkpoint

## Credits

- ASL-Citizen dataset and baseline code
- PyTorch
- OpenCV
- Ollama
- OBS Studio
