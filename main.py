import os
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import vgg16, efficientnet, xception
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).parent
MODELS_ROOT = PROJECT_ROOT / "models"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def safe_save_upload(file_storage) -> str:
    fname = secure_filename(file_storage.filename)
    out = UPLOAD_FOLDER / fname
    file_storage.save(str(out))
    return str(out)

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

class ModelBundle:
    
    def __init__(self, name: str, base_path: Path):
        self.name = name
        self.base = base_path / name
        self.meta = load_json(self.base / "meta.json")
        self.input_size = tuple(self.meta.get("input_size", [128, 128]))
        self.preprocessing = self.meta.get("preprocessing", "rescale_1/255")
        self.class_labels = self.meta.get("class_labels", [])
        self.last_conv_layer = self.meta.get("last_conv_layer", None)

        self.signature_model = None
        self.signature_input_key = None
        self.signature = None

        self.keras_model = None

        self._load_models()

    def _load_models(self):
        sig_path = self.base / "saved_model"
        if not sig_path.exists():
            sig_path = self.base / "exported_model"
        try:
            if sig_path.exists():
                self.signature_model = tf.saved_model.load(str(sig_path))
                sigs = getattr(self.signature_model, "signatures", {})
                if "serving_default" in sigs:
                    self.signature = sigs["serving_default"]
                elif "serve" in sigs:
                    self.signature = sigs["serve"]
                elif len(sigs) > 0:
                    self.signature = list(sigs.values())[0]
                if self.signature is not None:
                    try:
                        inputs = self.signature.structured_input_signature[1]
                        if isinstance(inputs, dict) and len(inputs) > 0:
                            self.signature_input_key = list(inputs.keys())[0]
                    except Exception:
                        self.signature_input_key = None
        except Exception as e:
            print(f"[{self.name}] failed to load signature model:", e)

        keras_paths = [
            self.base / "model_for_gradcam.keras",
            self.base / "model_for_gradcam.h5",
            self.base / "model.h5",
        ]
        for p in keras_paths:
            try:
                if p.exists():
                    self.keras_model = tf.keras.models.load_model(str(p), compile=False)
                    print(f"[{self.name}] loaded keras model from {p}")
                    break
            except Exception as e:
                print(f"[{self.name}] keras load failed for {p}: {e}")

    def preprocess(self, image_path: str) -> np.ndarray:
        img = load_img(image_path, target_size=self.input_size)
        arr = img_to_array(img).astype("float32")
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if self.preprocessing == "vgg16_preprocess":
            arr = vgg16.preprocess_input(arr)
        elif self.preprocessing == "efficientnet_preprocess":
            arr = efficientnet.preprocess_input(arr)
        elif self.preprocessing == "xception_preprocess":
            arr = xception.preprocess_input(arr)
        else:
            arr = arr / 255.0
        return np.expand_dims(arr, 0)

    def predict(self, image_path: str, binary_threshold: float = 0.5) -> Tuple[int, float, np.ndarray]:
        x = self.preprocess(image_path)
        preds = None

        if self.signature is not None:
            tf_x = tf.constant(x, dtype=tf.float32)
            try:
                if self.signature_input_key:
                    out = self.signature(**{self.signature_input_key: tf_x})
                else:
                    out = self.signature(tf_x)
            except Exception:
                out = self.signature(tf_x)

            if isinstance(out, dict):
                preds = list(out.values())[0].numpy()
            else:
                try:
                    preds = out.numpy()
                except Exception:
                    preds = np.asarray(out)
        elif self.keras_model is not None:
            preds = self.keras_model.predict(x)
        else:
            raise RuntimeError("No model available for prediction")

        preds = np.asarray(preds)

        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0, 0])
            idx = 1 if prob >= binary_threshold else 0
            conf = prob if idx == 1 else (1.0 - prob)
        elif preds.ndim == 2 and preds.shape[1] > 1:
            idx = int(np.argmax(preds, axis=1)[0])
            conf = float(np.max(preds, axis=1)[0])
        else:
            flat = np.ravel(preds)
            if flat.size == 1:
                prob = float(flat[0])
                idx = 1 if prob >= binary_threshold else 0
                conf = prob if idx == 1 else (1.0 - prob)
            else:
                idx = int(np.argmax(flat))
                conf = float(np.max(flat))

        return idx, conf, preds


    def find_last_conv(self):
        if self.last_conv_layer and self.keras_model:
            try:
                _ = self.keras_model.get_layer(self.last_conv_layer)
                return self.last_conv_layer
            except Exception:
                pass
        if not self.keras_model:
            raise RuntimeError("No keras_model loaded for Grad-CAM")
        for layer in reversed(self.keras_model.layers):
            if "conv" in layer.name or layer.__class__.__name__.lower().find("conv") >= 0:
                return layer.name
        raise RuntimeError("Could not find conv layer for Grad-CAM")

    def make_vanilla_gradcam_heatmap(self, img_array: np.ndarray, pred_index: Optional[int] = None) -> np.ndarray:
        last_conv = self.find_last_conv()
        last_conv_layer = self.keras_model.get_layer(last_conv)
        grad_model = tf.keras.models.Model(self.keras_model.inputs, [last_conv_layer.output, self.keras_model.output])
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[-1]
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            loss = predictions[:, pred_index]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("vanilla Grad-CAM: gradients are None.")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        for i in range(pooled_grads_np.shape[-1]):
            conv_outputs_np[..., i] *= pooled_grads_np[i]
        heatmap = np.mean(conv_outputs_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (self.input_size[1], self.input_size[0]))
        return heatmap

    def make_gradcam_heatmap(self, img_array: np.ndarray, pred_index: Optional[int] = None) -> np.ndarray:
        if self.keras_model is None:
            raise RuntimeError("Keras model not loaded for Grad-CAM++")

        last_conv = self.find_last_conv()
        last_conv_layer = self.keras_model.get_layer(last_conv)

        grad_model = tf.keras.models.Model(self.keras_model.inputs, [last_conv_layer.output, self.keras_model.output])
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[-1]

            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            else:
                pred_index = int(pred_index)

            try:
                tape.watch(conv_outputs)
            except Exception:
                pass

            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("Grad-CAM++: first-order grads are None, falling back to vanilla Grad-CAM.")
            try:
                return self.make_vanilla_gradcam_heatmap(img_array, pred_index=pred_index)
            except Exception as e:
                raise RuntimeError("Grad-CAM++ failed and vanilla fallback also failed: " + str(e))

        grads2 = tape.gradient(grads, conv_outputs)
        grads3 = None
        if grads2 is not None:
            grads3 = tape.gradient(grads2, conv_outputs)

        try:
            del tape
        except Exception:
            pass

        if grads2 is None or grads3 is None:
            print("Grad-CAM++: higher-order grads are None, falling back to vanilla Grad-CAM.")
            return self.make_vanilla_gradcam_heatmap(img_array, pred_index=pred_index)

        conv_outputs_np = conv_outputs[0].numpy()
        grads_np = grads[0].numpy()
        grads2_np = grads2[0].numpy()
        grads3_np = grads3[0].numpy()

        eps = 1e-8
        denom = (2.0 * grads2_np) + (conv_outputs_np * grads3_np)
        denom = np.where(np.abs(denom) < eps, eps, denom)
        alpha = grads2_np / denom

        positive_grads = np.maximum(grads_np, 0.0)

        weights = np.sum(alpha * positive_grads, axis=(0, 1))

        cam = np.sum(weights * conv_outputs_np, axis=-1)

        heatmap = np.maximum(cam, 0)
        maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1.0
        heatmap = heatmap / (maxv + 1e-12)

        heatmap = cv2.resize(heatmap, (self.input_size[1], self.input_size[0]))
        return heatmap

    def overlay_heatmap(self, orig_path: str, heatmap: np.ndarray, out_path: str, alpha: float = 0.4):
        img = Image.open(orig_path).convert("RGB")
        img = img.resize((heatmap.shape[1], heatmap.shape[0]))
        img_arr = np.array(img)
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
        overlay = colored_map * alpha + img_arr * (1 - alpha)
        overlay = np.uint8(overlay)
        Image.fromarray(overlay).save(out_path)
        return out_path

class ModelsRegistry:
    def __init__(self, models_root: Path):
        self.root = models_root
        self.bundles: Dict[str, ModelBundle] = {}
        self.scan_models()

    def scan_models(self):
        for p in sorted(self.root.iterdir()):
            if p.is_dir():
                try:
                    name = p.name
                    self.bundles[name] = ModelBundle(name, self.root)
                    print(f"[registry] loaded model bundle: {name}")
                except Exception as e:
                    print(f"[registry] failed to load {p}: {e}")

    def get_names(self):
        return list(self.bundles.keys())

    def get(self, name: str) -> ModelBundle:
        if name not in self.bundles:
            raise KeyError(f"Model '{name}' not found")
        return self.bundles[name]

registry = ModelsRegistry(MODELS_ROOT)

@app.route("/", methods=["GET", "POST"])
def index():
    models = registry.get_names()
    selected_model = models[0] if models else None
    return render_template("index.html", models=models, result=None, selected_model=selected_model)

@app.route("/predict", methods=["POST"])
def predict_route():
    model_name = request.form.get("model_name")
    if not model_name:
        return render_template("index.html", models=registry.get_names(), result="No model selected", selected_model=None)
    if "file" not in request.files:
        return render_template("index.html", models=registry.get_names(), result="No file part", selected_model=None)
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", models=registry.get_names(), result="No file selected", selected_model=None)
    if not allowed_file(file.filename):
        return render_template("index.html", models=registry.get_names(), result="Invalid file type", selected_model=None)
    saved_path = safe_save_upload(file)
    bundle = registry.get(model_name)
    try:
        idx, conf, raw_preds = bundle.predict(saved_path)
        label = bundle.class_labels[idx] if idx < len(bundle.class_labels) else f"idx_{idx}"
        if model_name == "Brain MRI Model":
            text = "No Tumor" if label == "notumor" else f"Tumor: {label}"
        elif model_name == "Chest X-Ray Model":
            text = "Normal" if label == "NORMAL" else "Pneumonia Detected"
        elif model_name == "Eye Retinal Scan Model":
            text = f"Retinal Condition: {label}"
        elif model_name == "Bone Fracture Model":
            text = "No Fracture" if label == "not fractured" else "Fracture Detected"
        overlay_url = None
        if bundle.keras_model is not None:
            try:
                arr = bundle.preprocess(saved_path)
                kpreds = bundle.keras_model.predict(arr)
                kidx = int(np.argmax(kpreds, axis=1)[0])
                heatmap = bundle.make_gradcam_heatmap(arr, pred_index=kidx)
                overlay_fname = f"overlay_{Path(saved_path).name}"
                overlay_path = UPLOAD_FOLDER / overlay_fname
                bundle.overlay_heatmap(saved_path, heatmap, str(overlay_path))
                overlay_url = f"/uploads/{overlay_fname}"
            except Exception as e:
                print("Grad-CAM failed:", e)
                traceback.print_exc()
        return render_template("index.html", models=registry.get_names(), result=text, confidence=f"{conf*100:.2f}%", file_path=f"/uploads/{Path(saved_path).name}", overlay_path=overlay_url, selected_model=model_name)
    except Exception as e:
        tb = traceback.format_exc()
        print("Prediction error:", tb)
        return render_template("index.html", models=registry.get_names(), result=f"Prediction error: {e}", selected_model=model_name)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


if __name__ == "__main__":
    app.run(debug=True)
