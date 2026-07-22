# utils

Helper modules for SPAAC (Student Performance Analysis and Attention in Classroom). These are the building blocks the emotion/attention pipeline imports: dataset loading, image augmentation, preprocessing, drawing, and Grad-CAM visualization.

This is a Python package (`__init__.py` present). Nothing here runs a full pipeline on its own — the modules are imported by the top-level scripts in `../` (`face-rec-emotion.py`, `emotions.py`). Most of the code is adapted from the mini-XCEPTION face classification work; the SPAAC-specific part is the label mapping that reduces FER2013 emotions to "Attentive" vs "NOT ATTENTIVE".

## Modules

| File | What it holds |
|------|---------------|
| `datasets.py` | `DataManager` loads FER2013 (CSV), IMDB gender (`.mat`), or KDEF (image folder). Plus `get_labels`, `get_class_to_arg`, and train/val split helpers. |
| `data_augmentation.py` | `ImageGenerator` — a batch generator with saturation, brightness, contrast, lighting-noise, random crop, and horizontal/vertical flip. Yields Keras-style `{'input_1': ...}, {'predictions': ...}` batches. |
| `preprocessor.py` | Small wrappers: `preprocess_input` (scale to [-1, 1]), `_imread`/`_imresize` over OpenCV, `to_categorical`. |
| `inference.py` | Runtime drawing/detection helpers: load a Haar cascade, `detectMultiScale`, draw boxes and text, apply bounding-box offsets. |
| `visualizer.py` | Matplotlib helpers to tile faces into a mosaic and plot conv-layer weights / feature maps. |
| `grad_cam.py` | Grad-CAM and guided backprop to produce class-activation heatmaps showing which face regions drove a prediction. |

## The SPAAC twist

`get_labels("fer2013")` in `datasets.py` does not return the seven raw emotions. It collapses them into an attention signal:

```python
{
    0: "NOT ATTENTIVE",  # angry
    1: "NOT ATTENTIVE",  # disgust
    2: "NOT ATTENTIVE",  # fear
    3: "Attentive",      # happy
    4: "NOT ATTENTIVE",  # sad
    5: "Attentive",      # surprise
    6: "Attentive",      # neutral
}
```

So the emotion classifier is repurposed as a proxy for whether a student looks engaged. That mapping is the project's core idea, not a general emotion label set.

## Stack

Read straight off the imports (there is no `requirements.txt` in this folder):

- NumPy, SciPy, pandas
- OpenCV (`cv2`)
- Keras + TensorFlow (`grad_cam.py` uses TF1-era APIs — `tf.get_default_graph`, `K.image_dim_ordering`, `ops._gradient_registry`)
- matplotlib
- h5py

## Usage

Import as a package from the project root (`SPAAC/`). Example — load FER2013 and read a batch of augmented images:

```python
from utils.datasets import DataManager, split_data
from utils.data_augmentation import ImageGenerator
from utils.datasets import get_labels

data = DataManager("fer2013", dataset_path="../datasets/fer2013/fer2013.csv")
faces, emotions = data.get_data()
labels = get_labels("fer2013")   # -> Attentive / NOT ATTENTIVE mapping
```

Draw a detection at runtime (the pattern used by `face-rec-emotion.py`):

```python
from utils.inference import load_detection_model, detect_faces, draw_bounding_box, draw_text

detector = load_detection_model("haarcascade_frontalface_default.xml")
faces = detect_faces(detector, gray_image)
for face in faces:
    draw_bounding_box(face, rgb_image, (255, 0, 0))
```

## Notes

- Dataset paths default to `../datasets/...` and model paths to `../trained_models/...`, so scripts expect to be run from the project directory with those folders present.
- `grad_cam.py` and `visualizer.py` have `__main__` blocks that expect pickled `faces.pkl` / `emotions.pkl` and specific `.hdf5` model files — they're debugging entry points, not part of the normal flow.
- `data_augmentation.py` still has a TODO for non-bounding-box random crop; `_do_random_crop` and `do_random_rotation` are near-duplicate and only safe for classification.
- `_load_fer2013` calls `pd.get_dummies(...).as_matrix()`, a pandas API removed in modern versions — this code targets an older pandas/Keras/TF1 stack.

Learning/research project. The code is adapted from existing mini-XCEPTION face-classification utilities and wired to the attention-detection use case.
