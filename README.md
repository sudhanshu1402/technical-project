# SPAAC — Student Performance & Attention Analysis in Class

A webcam/video pipeline that identifies known students by face and reads their facial expression frame by frame, then logs each person, their emotion, and an attendance flag. Built as a college project; this repo is the phase-1 prototype.

## What it does

Point it at a classroom camera (or a test video) and for every detected face it:

1. Matches the face against a small set of enrolled students (7 people, one photo each in `images/`).
2. Crops the face, runs it through a Keras CNN trained on FER2013, and predicts an emotion.
3. Draws a bounding box + label on the frame and appends a row to a CSV: name, emotion probability, emotion mode over the last frames, attendance = 1, and a timestamp.

The pitch behind it: a teacher can't watch every student at once, so use the existing CCTV feed to flag who's present and who looks disengaged. It's an academic proof of concept, not a deployed product.

## Two entry points

Both live in `SPAAC/` and must be run from inside that folder (paths are relative to it).

| Script | What it does |
| :--- | :--- |
| `face-rec-emotion.py` | The full thing: face recognition + emotion + facial-landmark dots + CSV logging. Webcam on by default (`USE_WEBCAM = True`). |
| `emotions.py` | Emotion detection only, no identity, no CSV. Simpler. Reads from `./test/testvdo.mp4` by default (`USE_WEBCAM = False`). |

`utils/` holds the supporting code: `datasets.py` (label maps + FER2013/IMDB/KDEF loaders), `inference.py` (box/text drawing, offsets), `preprocessor.py` (normalize to [-1, 1]), plus `grad_cam.py`, `visualizer.py`, and `data_augmentation.py` that aren't wired into the two entry scripts.

## Stack

- Python
- OpenCV (`cv2`) — capture, color conversion, drawing, window display
- `dlib` frontal face detector + the `face_recognition` library (dlib-based) for identity matching
- Keras / TensorFlow — loads `models/emotion_model.hdf5` (~850 KB, 48×48 grayscale input, FER2013-style mini-CNN)
- `imutils`, NumPy, pandas, matplotlib

There's no `requirements.txt`. Install what the imports need:

```bash
pip install opencv-python dlib face_recognition keras tensorflow imutils numpy pandas matplotlib scipy
```

`dlib` needs CMake and a C++ toolchain to build — that's usually the painful part of setup.

## Run

```bash
cd SPAAC
python face-rec-emotion.py     # face recognition + emotion + CSV
# or
python emotions.py             # emotion only
```

Press `q` to quit the window. To feed a file instead of the webcam, flip `USE_WEBCAM` at the top of the script; it falls back to `./test/testvdo.mp4`.

## Enrolling faces

`face-rec-emotion.py` hardcodes the roster near the top — one `face_recognition.load_image_file(...)` call per student, pointing at a JPG in `images/`, collected into `known_face_encodings` / `known_face_names`. To add or change students, edit those lists and drop matching photos in `images/`. Anyone not matched shows up as `UNKNOWN`.

## Things worth knowing (and rough edges)

This is old prototype code and it shows:

- **The CSV path is hardcoded to a Windows machine**: `C:\Users\admin\PycharmProjects\SPAAC\file.csv`. On any other machine `df.to_csv(...)` will fail. Change that line before expecting output.
- **The label map doesn't match the emotion strings.** `get_labels("fer2013")` in `datasets.py` returns `"Attentive"` / `"NOT ATTENTIVE"` (repurposed from the raw emotion classes), but both scripts branch on `"ANGRY"`, `"HAPPY"`, `"SAD"`, etc. Those comparisons never hit, so the box color falls through to the default. `emotions.py` displays the attentive/not-attentive mode; the color logic is effectively dead. If you want real emotion words on screen, restore the standard FER2013 label map.
- `datasets.py` uses `pandas.get_dummies(...).as_matrix()`, removed in modern pandas — the FER2013 loader only runs on old pandas.
- Recognition runs on a half-scale frame for speed; some drawing code assumes quarter-scale (`*= 2` vs the 0.50 resize), a leftover from the original source.
- The model and much of `utils/` are adapted from the open-source face-classification work by oarriaga; the classroom/attendance layer on top is the project's own.

## Layout

```
SPAAC/
  face-rec-emotion.py   real-time recognition + emotion + logging
  emotions.py           emotion-only demo
  models/               emotion_model.hdf5
  images/               enrolled student photos (one JPG each)
  test/                 sample videos, GIFs, project report PDF
  utils/                datasets, inference, preprocessing, grad-cam, etc.
```

## Scope

Student project, first phase. It works as a demo on a handful of known faces and a test video; it is not tuned, benchmarked, or hardened, and the emotion labels currently render as an attentive/not-attentive flag rather than named emotions. Treat it as a learning exercise in stitching face recognition and a CNN classifier into one OpenCV loop.

## License

MIT. See `LICENSE`.
