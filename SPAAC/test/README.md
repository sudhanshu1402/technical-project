# test — SPAAC sample media

Test inputs and demo outputs for [SPAAC](../) (Student Performance Analysis and Attention in Classroom), the classroom emotion/attention recognition project.

There is no code in this folder. It holds the sample videos fed to the face-recognition and emotion pipeline, the GIFs showing what the pipeline produces, and the written project report.

## Contents

| File | What it is |
|------|------------|
| `testvdo.mp4` (2.1 MB) | Sample classroom-style clip used as pipeline input |
| `dinner.mp4` (2.5 MB) | Second sample clip used as pipeline input |
| `testgif.gif` (5.0 MB) | Demo of the annotated output (faces + emotion labels) |
| `tested.gif` (1.1 MB) | Shorter/processed demo GIF |
| `report.pdf` (~957 KB) | 5-page project report (LaTeX) |

## How these are used

The processing code lives one level up in the SPAAC root:

- `../face-rec-emotion.py` — face detection + recognition + per-face emotion classification
- `../emotions.py` — emotion model / classification helper

Point those scripts at a video in this folder to see the recognizer run. See the [parent README](../README.md) for the full dependency list (OpenCV, dlib, face_recognition, Keras/TensorFlow, etc.) and how the pipeline fits together.

## Note

This is a coursework/academic project. The files here are demo material, not a distributable dataset — the clips are short samples, not a labeled training set.
