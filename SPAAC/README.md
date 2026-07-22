# SPAAC — Student Performance and Attention Analysis in Classroom

Real-time webcam tool that recognizes known students by face and classifies whether each one looks attentive, using a CNN trained on the FER2013 facial-emotion dataset.

This was a college group project. It's a working prototype, not production software — see the honest notes at the end before you try to run it.

## What it does

Point it at a classroom camera (or a video file) and, per frame, it:

1. Detects faces with dlib's frontal face detector.
2. Matches each face against a small gallery of enrolled students (`images/*.jpg`) using the `face_recognition` library.
3. Crops each face, runs it through a Keras CNN (`models/emotion_model.hdf5`), and maps the predicted FER2013 emotion to an attention label.
4. Draws a bounding box + label (`<name> is <emotion>`) on the video and logs a row per detection to a CSV with name, probability, attention mode, attendance flag, and timestamp.

The attention idea: FER2013's seven emotions are collapsed into two classes in `utils/datasets.py` — `happy`, `surprise`, `neutral` count as **Attentive**, everything else (`angry`, `disgust`, `fear`, `sad`) as **NOT ATTENTIVE**.

## Files

| File | Role |
|------|------|
| `face-rec-emotion.py` | Main script. Face recognition + emotion/attention + per-frame CSV logging. |
| `emotions.py` | Simpler variant: emotion detection only, no face recognition, reads from `test/testvdo.mp4`. |
| `utils/datasets.py` | FER2013/IMDB/KDEF dataset loaders and the emotion→attention label map (`get_labels`). |
| `utils/inference.py` | Drawing helpers (bounding box, text) and face-detection wrappers. |
| `utils/preprocessor.py` | Pixel normalization for model input. |
| `utils/grad_cam.py`, `visualizer.py`, `data_augmentation.py` | Training/visualization helpers (not used by the two run scripts). |
| `models/emotion_model.hdf5` | Pre-trained Keras emotion CNN (48×48 grayscale input). |
| `images/*.jpg` | Enrolled student face gallery. |
| `test/` | Sample videos, a demo GIF, and the project report PDF. |

## Stack

Python, with:

- OpenCV (`cv2`) — video capture, drawing, color conversion
- dlib — frontal face detector
- `face_recognition` — face encoding and matching (needs dlib + CMake to build)
- Keras / TensorFlow — loads and runs the emotion CNN
- NumPy, pandas — arrays and CSV output
- imutils — `face_utils.rect_to_bb`
- matplotlib — used by some util helpers

There's no `requirements.txt` in the repo. Install roughly:

```bash
pip install opencv-python dlib face_recognition keras tensorflow numpy pandas imutils matplotlib scipy
```

`dlib` needs CMake and a C++ toolchain installed first.

## Run

```bash
# Main version: face recognition + attention, uses webcam
python face-rec-emotion.py

# Emotion-only version, plays test/testvdo.mp4
python emotions.py
```

Press `q` to quit the video window.

Toggle the source with the `USE_WEBCAM` flag near the top of each script (`True` = webcam index 0, `False` = the video file in `test/`).

To enroll students, drop a clear front-facing `name.jpg` in `images/` and add the load/encode lines plus the name to `known_face_encodings` / `known_face_names` in `face-rec-emotion.py`.

## Known rough edges

Being honest about the prototype state:

- **Hardcoded CSV path.** `face-rec-emotion.py` writes to `C:\Users\admin\PycharmProjects\SPAAC\file.csv`. Change that line or logging will fail on any other machine.
- **Enrollment is hardcoded.** The seven students are loaded by name in the script rather than by scanning the `images/` folder.
- **The color-by-emotion branches never fire as written.** `get_labels("fer2013")` returns `Attentive` / `NOT ATTENTIVE`, but the `if emotion_text == "ANGRY"` etc. checks compare against raw emotion names, so the box color always falls through to the default green. Cosmetic, not fatal.
- **`datasets.py` uses `pd.get_dummies(...).as_matrix()`**, removed in modern pandas — only matters if you retrain, not for inference.
- The bottom display loop in `face-rec-emotion.py` reads `face_names` (empty) rather than the per-frame names, so that second label pass draws nothing.

The `utils/` code and the general architecture follow the well-known open-source face/emotion classification approach; the classroom attention framing, student gallery, and CSV attendance logging are what this project added on top.
