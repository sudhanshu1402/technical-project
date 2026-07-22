# models/

Pretrained weights for the facial-emotion classifier used by SPAAC.

## What's here

| File | Size | What it is |
|------|------|------------|
| `emotion_model.hdf5` | ~852 KB | A Keras/TensorFlow CNN trained on the FER2013 dataset. Saved in HDF5 format (full model: architecture + weights). |

This is the only artifact in the folder. There's no training code here — just the finished model that the SPAAC scripts load at runtime.

## What the model does

It's a 7-class facial-emotion classifier over the FER2013 emotion set (angry, disgust, fear, happy, sad, surprise, neutral). The small file size is consistent with a compact fully-convolutional architecture (mini-Xception style) rather than a large network.

Input/output contract, taken from how `emotions.py` and `face-rec-emotion.py` feed it:

- **Input**: a single-channel (grayscale) face crop, resized to the model's `input_shape` (48×48 for FER2013), normalized to the range `[-1, 1]`, shaped `(1, H, W, 1)`.
- **Output**: a probability vector over the 7 emotion classes. The calling code takes `argmax` for the label and `max` for confidence.

Normalization is done by `preprocess_input(x, v2=True)` in `../utils/preprocessor.py`:

```python
x = x.astype("float32") / 255.0
x = (x - 0.5) * 2.0   # -> [-1, 1]
```

## How it's loaded

Both SPAAC scripts reference it by relative path and load it directly with Keras:

```python
from keras.models import load_model

emotion_model_path = "./models/emotion_model.hdf5"
emotion_classifier = load_model(emotion_model_path)

# input size is read straight off the loaded model
emotion_target_size = emotion_classifier.input_shape[1:3]
```

Then, per detected face:

```python
gray_face = cv2.resize(gray_face, emotion_target_size)
gray_face = preprocess_input(gray_face, True)
gray_face = np.expand_dims(gray_face, 0)     # batch axis
gray_face = np.expand_dims(gray_face, -1)    # channel axis
pred = emotion_classifier.predict(gray_face)
label = emotion_labels[np.argmax(pred)]
```

Paths are resolved relative to the SPAAC project root (`../`), so run the scripts from there, not from inside `models/`.

## Note on the labels

SPAAC doesn't display the raw emotion names. `get_labels("fer2013")` in `../utils/datasets.py` remaps the 7 FER2013 class indices onto two attention states — happy/surprise/neutral become **Attentive**, the rest **NOT ATTENTIVE** — which is what actually gets drawn on screen and logged. The model itself still predicts the original 7 classes; the remap happens after `argmax`.

## Scope

Third-party pretrained weights bundled into a college project (Student Performance & Attention Analysis in Classroom). It was not trained as part of this repo, and there's no evaluation or benchmark data checked in here. Treat the accuracy as "whatever FER2013 mini-Xception gives you" — fine for a demo, not validated for real classroom use.
