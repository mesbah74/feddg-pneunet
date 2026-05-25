from __future__ import annotations

import json
import os
import traceback
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="The structure of `inputs` doesn't match the expected structure.*")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

tf.keras.mixed_precision.set_global_policy("float32")

IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5
DEFAULT_KNN_NEIGHBORS = 10
GRAPH_THRESHOLD = 0.3


@tf.keras.utils.register_keras_serializable(package="FedDGPneuNet")
class EdgeAwareGATv2(tf.keras.layers.Layer):
    def __init__(self, units: int, num_heads: int = 4, dropout_rate: float = 0.2, **kwargs: Any):
        super().__init__(**kwargs)
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)

    def build(self, input_shape: Any) -> None:
        feature_dim = int(input_shape[0][-1])
        self.W_l = self.add_weight(
            shape=(feature_dim, self.units * self.num_heads),
            initializer="glorot_uniform",
            trainable=True,
            name="W_l",
            dtype=tf.float32,
        )
        self.W_r = self.add_weight(
            shape=(feature_dim, self.units * self.num_heads),
            initializer="glorot_uniform",
            trainable=True,
            name="W_r",
            dtype=tf.float32,
        )
        self.a = self.add_weight(
            shape=(self.num_heads, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="a",
            dtype=tf.float32,
        )
        self.W_e = self.add_weight(
            shape=(self.num_heads,),
            initializer="glorot_uniform",
            trainable=True,
            name="W_e",
            dtype=tf.float32,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        x, adjacency = inputs
        x = tf.cast(x, tf.float32)
        adjacency = tf.cast(adjacency, tf.float32)
        batch = tf.shape(x)[0]
        nodes = tf.shape(x)[1]

        left = tf.reshape(tf.matmul(x, tf.cast(self.W_l, tf.float32)), (batch, nodes, self.num_heads, self.units))
        right = tf.reshape(tf.matmul(x, tf.cast(self.W_r, tf.float32)), (batch, nodes, self.num_heads, self.units))
        attention_kernel = tf.cast(self.a, tf.float32)
        src_score = tf.einsum("bnhu,hu->bhn", left, attention_kernel)
        dst_score = tf.einsum("bnhu,hu->bhn", right, attention_kernel)

        attention = tf.nn.leaky_relu(tf.expand_dims(src_score, -1) + tf.expand_dims(dst_score, -2), alpha=0.2)
        adjacency_expanded = tf.expand_dims(adjacency, 1)
        edge_bias = adjacency_expanded * tf.reshape(tf.cast(self.W_e, tf.float32), [1, self.num_heads, 1, 1])
        attention = attention + edge_bias
        attention = attention + (1.0 - tf.cast(adjacency_expanded > 0, tf.float32)) * (-1e9)
        attention = tf.nn.softmax(attention, axis=-1)

        if training and self.dropout_rate > 0:
            attention = tf.nn.dropout(attention, rate=self.dropout_rate)

        left_transposed = tf.transpose(left, [0, 2, 1, 3])
        output = tf.reduce_mean(tf.matmul(attention, left_transposed), axis=1)
        output = output + tf.cast(self.bias, tf.float32)
        return tf.nn.elu(output)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Any, Any, int]:
        return (input_shape[0][0], input_shape[0][1], self.units)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate})
        return config


@dataclass
class ModelArtifacts:
    feature_extractor: tf.keras.Model
    gat_model: tf.keras.Model
    reference_features: np.ndarray
    reference_labels: np.ndarray
    threshold: float
    graph_nodes: int
    graph_feature_dim: int


def load_threshold(model_dir: Path) -> float:
    path = model_dir / "threshold.txt"
    if not path.is_file():
        return DEFAULT_THRESHOLD
    try:
        value = float(path.read_text(encoding="utf-8").strip())
        return value if 0.0 < value < 1.0 else DEFAULT_THRESHOLD
    except ValueError:
        return DEFAULT_THRESHOLD


def model_graph_shape(model: tf.keras.Model) -> tuple[int, int]:
    x_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    nodes = int(x_shape[1]) if x_shape[1] is not None else 65
    feature_dim = int(x_shape[2]) if x_shape[2] is not None else 512
    return nodes, feature_dim


def load_artifacts(model_dir: str | Path = "model") -> ModelArtifacts:
    model_dir = Path(model_dir)
    feature_extractor = tf.keras.models.load_model(model_dir / "feature_extractor.h5", compile=False)
    gat_model = tf.keras.models.load_model(
        model_dir / "feddg_gatnet_model.h5",
        custom_objects={"EdgeAwareGATv2": EdgeAwareGATv2},
        compile=False,
    )
    reference_features = np.load(model_dir / "reference_features.npy").astype(np.float32)
    reference_labels = np.load(model_dir / "reference_labels.npy").astype(np.float32)
    nodes, feature_dim = model_graph_shape(gat_model)
    return ModelArtifacts(
        feature_extractor=feature_extractor,
        gat_model=gat_model,
        reference_features=reference_features,
        reference_labels=reference_labels,
        threshold=load_threshold(model_dir),
        graph_nodes=nodes,
        graph_feature_dim=feature_dim,
    )


def extractor_has_internal_rescaling(model: tf.keras.Model) -> bool:
    return any(layer.__class__.__name__.lower() == "rescaling" for layer in model.layers[:8])


def preprocess_image(image: Image.Image, feature_extractor: tf.keras.Model) -> tuple[np.ndarray, Image.Image, str]:
    original = ImageOps.exif_transpose(image).convert("RGB")
    resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype(np.float32)
    if extractor_has_internal_rescaling(feature_extractor):
        normalization = "internal EfficientNetV2 rescaling"
    else:
        array = array / 255.0
        normalization = "manual 0-1 normalization"
    return np.expand_dims(array, axis=0), original, normalization


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-7)


def select_reference_nodes(
    query_feature: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    required_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(reference_features) < required_neighbors:
        raise ValueError(f"Reference bank has {len(reference_features)} rows but {required_neighbors} neighbors are required.")

    query_norm = l2_normalize_rows(query_feature.reshape(1, -1))[0]
    reference_norm = l2_normalize_rows(reference_features)
    similarities = reference_norm @ query_norm
    candidate_indices = np.argpartition(-similarities, required_neighbors - 1)[:required_neighbors]
    sorted_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
    return (
        reference_features[sorted_indices].astype(np.float32),
        reference_labels[sorted_indices].astype(np.float32),
        similarities[sorted_indices].astype(np.float32),
        sorted_indices.astype(np.int64),
    )


def build_dynamic_graph(features: np.ndarray, n_neighbors: int = DEFAULT_KNN_NEIGHBORS, threshold: float = GRAPH_THRESHOLD) -> np.ndarray:
    node_count = features.shape[0]
    if node_count == 1:
        return np.ones((1, 1), dtype=np.float32)
    effective_neighbors = min(max(1, n_neighbors), node_count - 1)
    normalized_features = normalize(features, norm="l2")
    similarity_matrix = np.dot(normalized_features, normalized_features.T).astype(np.float32)
    knn_mask = kneighbors_graph(features, n_neighbors=effective_neighbors, mode="connectivity", include_self=True).toarray().astype(np.float32)
    adjacency = similarity_matrix * knn_mask
    adjacency = np.where(adjacency > threshold, adjacency, 0).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    degree = np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-7)
    return (adjacency / degree).astype(np.float32)


def query_probability(output: np.ndarray) -> float:
    output = np.asarray(output)
    if output.ndim == 3:
        value = output[0, 0, 0]
    elif output.ndim == 2:
        value = output[0, 0]
    elif output.ndim == 1:
        value = output[0]
    else:
        value = output.reshape(-1)[0]
    return float(np.clip(value, 0.0, 1.0))


def query_probability_tensor(output: tf.Tensor) -> tf.Tensor:
    rank = len(output.shape)
    if rank == 3:
        return output[:, 0, 0]
    if rank == 2:
        return output[:, 0]
    if rank == 1:
        return output
    return tf.reshape(output, [-1])


def find_last_spatial_layer(model: tf.keras.Model) -> Optional[str]:
    for layer in reversed(model.layers):
        try:
            output = layer.output
            if isinstance(output, (list, tuple)):
                output = output[0]
            if len(output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    stops = np.array([0.0, 0.35, 0.65, 1.0], dtype=np.float32)
    colors = np.array([[13, 34, 92], [21, 166, 200], [255, 210, 74], [214, 69, 80]], dtype=np.float32)
    channels = [np.interp(heatmap, stops, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def generate_gradcam(
    artifacts: ModelArtifacts,
    image_tensor: np.ndarray,
    original_image: Image.Image,
    selected_reference_features: np.ndarray,
    adjacency: np.ndarray,
    output_path: Path,
) -> Optional[Path]:
    layer_name = find_last_spatial_layer(artifacts.feature_extractor)
    if layer_name is None:
        return None

    try:
        grad_model = tf.keras.Model(
            artifacts.feature_extractor.inputs,
            [artifacts.feature_extractor.get_layer(layer_name).output, artifacts.feature_extractor.output],
        )
        reference_tensor = tf.convert_to_tensor(selected_reference_features, dtype=tf.float32)
        adjacency_tensor = tf.convert_to_tensor(np.expand_dims(adjacency, axis=0), dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, query_features = grad_model(image_tensor, training=False)
            tape.watch(conv_outputs)
            query_features = tf.reshape(query_features, [tf.shape(query_features)[0], -1])
            graph_features = tf.concat([query_features, reference_tensor], axis=0)
            graph_features = tf.expand_dims(graph_features, axis=0)
            prediction = artifacts.gat_model([graph_features, adjacency_tensor], training=False)
            score = query_probability_tensor(prediction)[0]

        gradients = tape.gradient(score, conv_outputs)
        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_gradients, axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        max_value = float(np.max(heatmap))
        if max_value <= 1e-7:
            return None

        heatmap = heatmap / max_value
        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(original_image.size, Image.Resampling.BICUBIC)
        heatmap_resized = np.asarray(heatmap_image).astype(np.float32) / 255.0
        heatmap_color = colorize_heatmap(heatmap_resized)
        base = np.asarray(original_image.convert("RGB")).astype(np.float32)
        alpha = 0.22 + (0.48 * heatmap_resized[..., None])
        overlay = np.clip(base * (1.0 - alpha) + heatmap_color * alpha, 0, 255).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path, quality=92, optimize=True)
        return output_path
    except Exception:
        traceback.print_exc()
        return None


def load_report_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, fill: str, max_width: int) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + 10
    return y


def paste_report_image(page: Image.Image, draw: ImageDraw.ImageDraw, image_path: Optional[Path], box: tuple[int, int, int, int], title: str) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill="#f7fbff", outline="#d7e4f1", width=2)
    draw.text((x1 + 24, y1 + 18), title, font=load_report_font(28, True), fill="#111c2f")
    image_area = (x1 + 24, y1 + 70, x2 - 24, y2 - 24)
    if image_path is None or not image_path.is_file():
        draw.rounded_rectangle(image_area, radius=12, fill="#eaf3fb", outline="#d7e4f1")
        draw.text((image_area[0] + 24, image_area[1] + 28), "Image unavailable", font=load_report_font(22), fill="#63748d")
        return
    source = Image.open(image_path)
    source = ImageOps.exif_transpose(source).convert("RGB")
    fitted = ImageOps.contain(source, (image_area[2] - image_area[0], image_area[3] - image_area[1]), Image.Resampling.LANCZOS)
    px = image_area[0] + ((image_area[2] - image_area[0]) - fitted.width) // 2
    py = image_area[1] + ((image_area[3] - image_area[1]) - fitted.height) // 2
    page.paste(fitted, (px, py))


def create_prediction_report(image_path: Path, heatmap_path: Optional[Path], output_path: Path, payload: dict[str, Any]) -> Path:
    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(48, True)
    h2_font = load_report_font(30, True)
    body_font = load_report_font(24)
    small_font = load_report_font(20)
    label_font = load_report_font(21, True)

    draw.rectangle((0, 0, width, 170), fill="#0b4fa3")
    draw.rectangle((0, 160, width, 170), fill="#16a7c7")
    draw.text((70, 46), "FedDG-PneuNet", font=title_font, fill="#ffffff")
    draw.text((70, 108), "AI-Powered Pneumonia Detection Report", font=body_font, fill="#dff4ff")
    draw.text((850, 58), "Prediction Timestamp", font=small_font, fill="#dff4ff")
    draw.text((850, 90), payload["timestamp"], font=small_font, fill="#ffffff")

    draw.rounded_rectangle((70, 215, 1170, 430), radius=22, fill="#f7fbff", outline="#d7e4f1", width=2)
    draw.text((100, 248), "Prediction Result", font=h2_font, fill="#111c2f")
    result_color = "#d64550" if payload["prediction"] == "Pneumonia" else "#12a37c"
    draw.rounded_rectangle((100, 305, 430, 378), radius=20, fill=result_color)
    draw.text((128, 324), payload["prediction"], font=load_report_font(34, True), fill="#ffffff")
    draw.text((500, 292), "Confidence Score", font=label_font, fill="#63748d")
    draw.text((500, 326), f"{payload['confidence']:.2f}%", font=load_report_font(42, True), fill="#0b4fa3")
    draw.text((760, 292), "AI Model Information", font=label_font, fill="#63748d")
    draw.text((760, 326), "FedDG-GATNet", font=load_report_font(28, True), fill="#111c2f")
    draw.text((760, 365), "Federated Dynamic Graph Neural Network", font=small_font, fill="#63748d")

    paste_report_image(page, draw, image_path, (70, 480, 590, 960), "Uploaded Chest X-ray Image")
    paste_report_image(page, draw, heatmap_path, (650, 480, 1170, 960), "Grad-CAM Heatmap Image")

    draw.rounded_rectangle((70, 1015, 1170, 1195), radius=18, fill="#ffffff", outline="#d7e4f1", width=2)
    draw.text((100, 1048), "Report Details", font=h2_font, fill="#111c2f")
    draw.text((100, 1102), "Application Name: FedDG-PneuNet", font=body_font, fill="#111c2f")
    draw.text((100, 1142), f"Pneumonia Probability: {payload['probability'] * 100:.2f}%", font=body_font, fill="#111c2f")
    draw.text((650, 1102), f"Graph Nodes: {payload['graph_nodes']}", font=body_font, fill="#111c2f")
    draw.text((650, 1142), f"Reference Neighbors: {payload['neighbors']}", font=body_font, fill="#111c2f")

    draw.rounded_rectangle((70, 1250, 1170, 1455), radius=18, fill="#fff9ed", outline="#f0d8a8", width=2)
    draw.text((100, 1285), "Medical Disclaimer", font=h2_font, fill="#8a5a00")
    draw_wrapped_text(
        draw,
        "This system is intended for research and educational purposes only. It does not replace professional medical diagnosis or treatment. Please consult a licensed healthcare professional for clinical evaluation.",
        (100, 1338),
        body_font,
        "#6e4a08",
        980,
    )

    draw.rectangle((0, height - 94, width, height), fill="#0f1d31")
    draw.text((70, height - 62), "FedDG-PneuNet Medical AI Research Platform", font=small_font, fill="#ffffff")
    draw.text((760, height - 62), "Generated for educational and research use", font=small_font, fill="#c7d7e8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, "PDF", resolution=150.0)
    return output_path


def run_prediction(uploaded_file: Any, artifacts: ModelArtifacts, uploads_dir: str | Path = "uploads", reports_dir: str | Path = "reports") -> dict[str, Any]:
    uploads_dir = Path(uploads_dir)
    reports_dir = Path(reports_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    image_path = uploads_dir / f"xray_{run_id}{suffix}"

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.save(image_path)

    image_tensor, original_image, normalization_mode = preprocess_image(image, artifacts.feature_extractor)
    query_feature = artifacts.feature_extractor.predict(image_tensor, verbose=0).astype(np.float32)[0]
    if query_feature.shape[0] != artifacts.graph_feature_dim:
        raise ValueError(f"Feature mismatch: extractor returned {query_feature.shape[0]}, graph model expects {artifacts.graph_feature_dim}.")

    required_neighbors = artifacts.graph_nodes - 1
    selected_features, selected_labels, similarities, selected_indices = select_reference_nodes(
        query_feature,
        artifacts.reference_features,
        artifacts.reference_labels,
        required_neighbors,
    )
    graph_features = np.vstack([query_feature.reshape(1, -1), selected_features]).astype(np.float32)
    adjacency = build_dynamic_graph(graph_features)
    model_output = artifacts.gat_model.predict([np.expand_dims(graph_features, 0), np.expand_dims(adjacency, 0)], verbose=0)
    pneumonia_probability = query_probability(model_output)
    prediction = "Pneumonia" if pneumonia_probability >= artifacts.threshold else "Normal"
    confidence = pneumonia_probability if prediction == "Pneumonia" else (1.0 - pneumonia_probability)

    heatmap_path = generate_gradcam(
        artifacts=artifacts,
        image_tensor=image_tensor,
        original_image=original_image,
        selected_reference_features=selected_features,
        adjacency=adjacency,
        output_path=reports_dir / f"gradcam_{run_id}.jpg",
    )

    payload: dict[str, Any] = {
        "ok": True,
        "application": "FedDG-PneuNet",
        "prediction": prediction,
        "confidence": round(float(confidence * 100.0), 2),
        "probability": round(float(pneumonia_probability), 6),
        "threshold": round(float(artifacts.threshold), 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "graph_nodes": artifacts.graph_nodes,
        "neighbors": required_neighbors,
        "normal_neighbors": int(np.sum(selected_labels < 0.5)),
        "pneumonia_neighbors": int(np.sum(selected_labels >= 0.5)),
        "normalization": normalization_mode,
        "top_similarity": round(float(similarities[0]), 6) if len(similarities) else None,
        "top_reference_index": int(selected_indices[0]) if len(selected_indices) else None,
        "image_path": str(image_path),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
    }

    report_path = create_prediction_report(image_path, heatmap_path, reports_dir / f"report_{run_id}.pdf", payload)
    payload["report_path"] = str(report_path)
    prediction_path = reports_dir / f"prediction_{run_id}.json"
    prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["saved_prediction_path"] = str(prediction_path)
    return payload

