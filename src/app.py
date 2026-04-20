from pathlib import Path
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing_utils
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
MODELS_DIR = ASSETS_DIR / "models"

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Imágenes por expresión (memes adjuntos por el usuario)
EXPRESSION_IMAGES = {
    "surprise": "hampter_surprise.png",  # boca abierta / sorpresa (soyface)
    "cry": "hampter_cry.png",  # llanto / angustia (ceño + ojos)
    "pucker": "hampter_pucker.png",  # duck face / morritos
    "joy": "hampter_joy.png",  # felicidad / sonrisa amplia
    "confused": "hampter_confused.png",  # confusión / “brainlet”
    "neutral": "hampter_neutral.png",  # rostro neutro / leve sonrisa
}
HAND_GESTURE_IMAGES = {
    "heart_hands": "hands_heart.png",
    "open_palm": "open_hand.png",
    "peace": "Peace.png",
    "thumb_up": "thump_up.png",
    "thumb_down": "thump_down.png",
    "middle_finger": "hampter_middle_finger.png",
}

# Reglas fáciles de editar:
# - Se evalúan en orden.
# - Usa "any" para ignorar la parte de cara o mano.
# - Cuando me compartas nuevas expresiones, solo ajustamos esta tabla + imagen.
TRIGGER_RULES: list[tuple[str, str, str]] = [
    ("any", "heart_hands", "heart_hands"),
    ("any", "open_palm", "open_palm"),
    ("any", "peace", "peace"),
    ("any", "thumb_up", "thumb_up"),
    ("any", "thumb_down", "thumb_down"),
    ("any", "middle_finger", "middle_finger"),
    ("surprise", "open_palm", "surprise"),
    ("cry", "fist", "cry"),
    ("pucker", "peace", "pucker"),
    ("joy", "open_palm", "joy"),
    ("confused", "thumb_up", "confused"),
    ("neutral", "any", "neutral"),
    ("any", "open_palm", "joy"),
    ("any", "peace", "pucker"),
]
HAND_REQUIRED_GESTURES = {"heart_hands", "middle_finger", "thumb_down", "open_palm", "fist", "peace", "thumb_up"}
FACE_ONLY_MEMES = {"surprise", "cry", "pucker", "joy", "confused", "neutral"}

# Landmarks (Face Landmarker / mesh 468+)
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14

# Umbrales: más estrictos para reducir confusiones entre clases
MOUTH_OPEN_THRESHOLD = 0.040
MOUTH_OPEN_STRONG = 0.052
BLEND_JAW_OPEN = 0.38

SMILE_WIDTH_THRESHOLD = 0.105
SMILE_WIDTH_WITH_BLEND = 0.092
BLEND_SMILE = 0.40
BLEND_SMILE_WEAK = 0.22

BLEND_PUCKER = 0.42
BLEND_PUCKER_RELAXED = 0.30
PUCKER_MAX_MOUTH_WIDTH = 0.095
PUCKER_MAX_JAW_RATIO = 0.68
PUCKER_MAX_OPEN_RATIO = 0.92

# Tristeza / llanto (mouthFrown + ojos / cejas)
FROWN_CAP_FOR_SURPRISE = 0.30
CRY_FROWN_MIN = 0.30
CRY_FROWN_STRONG = 0.46
CRY_SQUINT = 0.13
CRY_BROW_DOWN = 0.11
CRY_SMILE_CAP = 0.22

# Confusión (ceja asimétrica o ceño interno alto sin sonrisa)
CONFUSED_BROW_ASYM = 0.19
CONFUSED_INNER_BROW = 0.36
CONFUSED_MAX_MOUTH_WIDTH = 0.098

SMOOTH_ALPHA = 0.58
HAND_STABLE_FRAMES = 3
MEME_STABLE_FRAMES = 3
SHOW_TRIGGER_WINDOW = False

HAND_TIP_IDS = [4, 8, 12, 16, 20]
HAND_PIP_IDS = [3, 6, 10, 14, 18]
HAND_WRIST_ID = 0
THUMB_MCP_ID = 2
MIDDLE_MCP_ID = 9
HEART_DISTANCE_THRESHOLD = 0.11
FOLDED_FINGER_MARGIN_RATIO = 0.18
THUMB_VERTICAL_RATIO = 1.25


def bgra_from_black_key(bgr_or_bgra: np.ndarray, black_threshold: int = 45) -> np.ndarray:
    if bgr_or_bgra.shape[2] == 4:
        return bgr_or_bgra
    b, g, r = cv2.split(bgr_or_bgra)
    brightness = cv2.max(cv2.max(b, g), r)
    alpha = np.where(brightness <= black_threshold, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r, alpha])


def load_expression_assets() -> dict[str, np.ndarray]:
    """Carga todas las imágenes de expresión desde BASE_DIR."""
    assets: dict[str, np.ndarray] = {}
    for key, filename in EXPRESSION_IMAGES.items():
        path = IMAGES_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Falta la imagen {path}")
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"No se pudo leer {path}")
        assets[key] = bgra_from_black_key(raw)
    return assets


def load_optional_hand_assets() -> dict[str, np.ndarray]:
    """Carga memes de gestos de mano si existen en disco."""
    assets: dict[str, np.ndarray] = {}
    for key, filename in HAND_GESTURE_IMAGES.items():
        path = IMAGES_DIR / filename
        if not path.is_file():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        assets[key] = bgra_from_black_key(raw)
    return assets


def blendshape_dict(blendshapes_per_face) -> dict[str, float]:
    if not blendshapes_per_face or not blendshapes_per_face[0]:
        return {}
    out: dict[str, float] = {}
    for c in blendshapes_per_face[0]:
        if c.category_name and c.score is not None:
            key = c.category_name.lower().replace(" ", "").replace("_", "")
            out[key] = float(c.score)
    return out


def bget(blend: dict[str, float], *names: str) -> float:
    """Lee blendshape probando variantes de nombre (MediaPipe puede variar)."""
    best = 0.0
    for n in names:
        k = n.lower().replace(" ", "").replace("_", "")
        best = max(best, blend.get(k, 0.0))
    return best


def landmark_scores(landmarks: list) -> tuple[float, float]:
    """Retorna (apertura_boca vertical, ancho_boca entre comisuras) normalizado."""
    mouth_open = abs(landmarks[LOWER_LIP_CENTER].y - landmarks[UPPER_LIP_CENTER].y)
    dx = landmarks[MOUTH_LEFT_CORNER].x - landmarks[MOUTH_RIGHT_CORNER].x
    dy = landmarks[MOUTH_LEFT_CORNER].y - landmarks[MOUTH_RIGHT_CORNER].y
    mouth_width = float(np.hypot(dx, dy))
    return mouth_open, mouth_width


def classify_hand_gesture(hand_landmarks: list, is_right_like: bool) -> str:
    """Clasifica gestos simples: open_palm, fist, peace, thumb_up, one, two."""
    pts = hand_landmarks
    if not isinstance(pts, list) or len(pts) < 21:
        return "unknown"
    fingers_up = []

    wrist = pts[HAND_WRIST_ID]
    middle_mcp = pts[MIDDLE_MCP_ID]
    palm_scale = max(float(np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)), 1e-6)
    fold_margin = FOLDED_FINGER_MARGIN_RATIO * palm_scale

    # Pulgar: usa X considerando si es mano derecha/izquierda aproximada.
    thumb_tip = pts[HAND_TIP_IDS[0]]
    thumb_pip = pts[HAND_PIP_IDS[0]]
    thumb_mcp = pts[THUMB_MCP_ID]
    thumb_extended = thumb_tip.x > thumb_pip.x if is_right_like else thumb_tip.x < thumb_pip.x
    thumb_vertical_up = thumb_tip.y < (thumb_pip.y - fold_margin)
    thumb_vertical_down = thumb_tip.y > (thumb_pip.y + fold_margin)
    thumb_vertical_span = abs(thumb_tip.y - thumb_mcp.y)
    thumb_horizontal_span = abs(thumb_tip.x - thumb_mcp.x)
    thumb_is_mostly_vertical = thumb_vertical_span > thumb_horizontal_span * THUMB_VERTICAL_RATIO
    # Para pulgar usamos señal combinada (lateral o vertical) para robustez.
    thumb_is_up_candidate = thumb_extended or thumb_vertical_up
    fingers_up.append(thumb_is_up_candidate)

    # Índice, medio, anular y meñique: punta más arriba (y menor) que su PIP.
    for tip_id, pip_id in zip(HAND_TIP_IDS[1:], HAND_PIP_IDS[1:]):
        fingers_up.append(pts[tip_id].y < (pts[pip_id].y - fold_margin))

    count_up = sum(fingers_up)
    index_up = fingers_up[1]
    middle_up = fingers_up[2]
    ring_up = fingers_up[3]
    pinky_up = fingers_up[4]
    non_thumb_folded = not index_up and not middle_up and not ring_up and not pinky_up

    if count_up == 0:
        return "fist"
    if middle_up and not index_up and not ring_up and not pinky_up:
        return "middle_finger"
    if (
        non_thumb_folded
        and thumb_vertical_down
        and (thumb_extended or thumb_is_mostly_vertical)
        and thumb_is_mostly_vertical
    ):
        return "thumb_down"
    if (
        non_thumb_folded
        and thumb_vertical_up
        and (thumb_extended or thumb_is_mostly_vertical)
        and thumb_is_mostly_vertical
    ):
        return "thumb_up"
    if count_up >= 4:
        return "open_palm"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "one"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "two"
    return "unknown"


def detect_heart_hands(two_hands: list[list]) -> bool:
    """Heurística simple para gesto de corazón con dos manos."""
    if len(two_hands) < 2:
        return False
    a = two_hands[0]
    b = two_hands[1]
    if len(a) < 21 or len(b) < 21:
        return False

    a_thumb = a[4]
    a_index = a[8]
    b_thumb = b[4]
    b_index = b[8]

    thumbs_close = float(np.hypot(a_thumb.x - b_thumb.x, a_thumb.y - b_thumb.y)) < HEART_DISTANCE_THRESHOLD
    indexes_close = float(np.hypot(a_index.x - b_index.x, a_index.y - b_index.y)) < HEART_DISTANCE_THRESHOLD
    vertical_order = (a_index.y + b_index.y) * 0.5 < (a_thumb.y + b_thumb.y) * 0.5
    return thumbs_close and indexes_close and vertical_order


def resolve_trigger(face_expression: str, hand_gesture: str) -> str:
    hand_visible = hand_gesture not in {"none", "unknown"}
    if not hand_visible:
        # Sin mano detectada: solo memes de expresión facial.
        return face_expression if face_expression in FACE_ONLY_MEMES else "neutral"

    for f_rule, h_rule, target in TRIGGER_RULES:
        if h_rule in HAND_REQUIRED_GESTURES and not hand_visible:
            continue
        face_ok = f_rule == "any" or f_rule == face_expression
        hand_ok = h_rule == "any" or h_rule == hand_gesture
        if face_ok and hand_ok:
            return target
    return "neutral"


def pick_meme_asset(
    meme_assets: dict[str, np.ndarray],
    hand_assets: dict[str, np.ndarray],
    active_meme: str,
    expression: str,
    hand_gesture: str,
) -> np.ndarray:
    # Si es meme de mano y existe su asset dedicado, úsalo.
    if hand_gesture in HAND_REQUIRED_GESTURES and active_meme in hand_assets:
        return hand_assets[active_meme]

    # Si falta asset dedicado de mano, cae a la expresión facial vigente.
    if hand_gesture in HAND_REQUIRED_GESTURES and expression in meme_assets:
        return meme_assets[expression]

    return meme_assets.get(active_meme, meme_assets["neutral"])


class TemporalLabelSmoother:
    """Confirma una etiqueta solo cuando aparece N frames seguidos."""

    def __init__(self, stable_frames: int, initial: str, immediate_labels: set[str] | None = None) -> None:
        self._stable_frames = max(1, stable_frames)
        self._candidate = initial
        self._count = 0
        self._confirmed = initial
        self._immediate = immediate_labels or set()

    def update(self, new_label: str) -> str:
        if new_label in self._immediate:
            self._candidate = new_label
            self._count = self._stable_frames
            self._confirmed = new_label
            return self._confirmed

        if new_label == self._candidate:
            self._count += 1
        else:
            self._candidate = new_label
            self._count = 1

        if self._count >= self._stable_frames:
            self._confirmed = self._candidate
        return self._confirmed


class ExpressionSmoother:
    def __init__(self) -> None:
        self._mouth_open = 0.0
        self._mouth_width = 0.0
        self._pucker = 0.0
        self._frown = 0.0
        self._squint = 0.0
        self._brow_down = 0.0
        self._brow_asym = 0.0
        self._brow_inner = 0.0

    def update(
        self,
        mouth_open: float,
        mouth_width: float,
        blend: dict[str, float],
    ) -> str:
        a = SMOOTH_ALPHA
        self._mouth_open = a * mouth_open + (1.0 - a) * self._mouth_open
        self._mouth_width = a * mouth_width + (1.0 - a) * self._mouth_width

        jaw = bget(blend, "jawOpen", "jaw_open")
        puck_raw = bget(blend, "mouthPucker", "mouth_pucker")
        self._pucker = a * puck_raw + (1.0 - a) * self._pucker

        frown_l = bget(blend, "mouthFrownLeft", "mouth_frown_left")
        frown_r = bget(blend, "mouthFrownRight", "mouth_frown_right")
        frown_avg = 0.5 * (frown_l + frown_r)
        self._frown = a * frown_avg + (1.0 - a) * self._frown

        squint_l = bget(blend, "eyeSquintLeft", "eye_squint_left")
        squint_r = bget(blend, "eyeSquintRight", "eye_squint_right")
        squint_avg = 0.5 * (squint_l + squint_r)
        self._squint = a * squint_avg + (1.0 - a) * self._squint

        bd_l = bget(blend, "browDownLeft", "brow_down_left")
        bd_r = bget(blend, "browDownRight", "brow_down_right")
        brow_down_avg = 0.5 * (bd_l + bd_r)
        self._brow_down = a * brow_down_avg + (1.0 - a) * self._brow_down

        bo_l = bget(blend, "browOuterUpLeft", "brow_outer_up_left")
        bo_r = bget(blend, "browOuterUpRight", "brow_outer_up_right")
        brow_asym = abs(bo_l - bo_r)
        self._brow_asym = a * brow_asym + (1.0 - a) * self._brow_asym

        biu = bget(blend, "browInnerUp", "brow_inner_up")
        self._brow_inner = a * biu + (1.0 - a) * self._brow_inner

        smile_b = max(
            bget(blend, "mouthSmileLeft", "mouth_smile_left"),
            bget(blend, "mouthSmileRight", "mouth_smile_right"),
        )

        # 1) Sorpresa: boca muy abierta pero sin ceño de “llanto” fuerte
        strong_open = self._mouth_open >= MOUTH_OPEN_STRONG
        jaw_and_open = jaw >= BLEND_JAW_OPEN and self._mouth_open >= MOUTH_OPEN_THRESHOLD * 0.88
        if (strong_open or jaw_and_open) and self._frown < FROWN_CAP_FOR_SURPRISE:
            return "surprise"

        # 2) Llanto / angustia: ceño labial + (ojos entrecerrados o cejas bajas) o ceño muy fuerte
        distress = self._squint >= CRY_SQUINT or self._brow_down >= CRY_BROW_DOWN
        cry_ok = (
            self._frown >= CRY_FROWN_MIN
            and smile_b < CRY_SMILE_CAP
            and (distress or self._frown >= CRY_FROWN_STRONG)
        )
        if cry_ok:
            return "cry"

        # 3) Morritos: mouthPucker, sin señal de sorpresa
        pucker_tight = self._pucker >= BLEND_PUCKER
        pucker_loose = (
            self._pucker >= BLEND_PUCKER_RELAXED
            and self._mouth_width <= PUCKER_MAX_MOUTH_WIDTH
            and jaw <= BLEND_JAW_OPEN * PUCKER_MAX_JAW_RATIO
            and self._mouth_open <= MOUTH_OPEN_THRESHOLD * PUCKER_MAX_OPEN_RATIO
        )
        if pucker_tight or pucker_loose:
            return "pucker"

        # 4) Alegría: ancho real + blend de sonrisa (evita falsos positivos)
        wide_mouth = self._mouth_width >= SMILE_WIDTH_THRESHOLD
        medium_mouth = self._mouth_width >= SMILE_WIDTH_WITH_BLEND
        if wide_mouth and smile_b >= BLEND_SMILE_WEAK:
            return "joy"
        if medium_mouth and smile_b >= BLEND_SMILE:
            return "joy"

        # 5) Confusión / escepticismo: asimetría de cejas o ceño interno alto con boca poco sonriente
        if self._brow_asym >= CONFUSED_BROW_ASYM and smile_b < 0.30:
            return "confused"
        if (
            self._brow_inner >= CONFUSED_INNER_BROW
            and smile_b < 0.24
            and self._mouth_width <= CONFUSED_MAX_MOUTH_WIDTH
        ):
            return "confused"

        return "neutral"


def overlay_meme(frame: np.ndarray, meme_bgra: np.ndarray, scale: float = 0.32) -> np.ndarray:
    h, w = frame.shape[:2]
    target_w = max(1, int(w * scale))
    aspect = meme_bgra.shape[0] / max(meme_bgra.shape[1], 1)
    target_h = max(1, int(target_w * aspect))

    resized = cv2.resize(meme_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

    x1 = w - target_w - 12
    y1 = h - target_h - 12
    if x1 < 0 or y1 < 0:
        return frame

    alpha = resized[:, :, 3] / 255.0
    alpha = alpha[:, :, np.newaxis]
    bgr = resized[:, :, :3]
    y2 = y1 + target_h
    x2 = x1 + target_w
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (alpha * bgr + (1.0 - alpha) * roi).astype(np.uint8)
    return frame


def ensure_face_landmarker_model() -> Path:
    path = MODELS_DIR / "face_landmarker.task"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 500_000:
        return path
    print("Descargando face_landmarker.task (solo la primera vez)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    return path


def ensure_hand_landmarker_model() -> Path:
    path = MODELS_DIR / "hand_landmarker.task"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 500_000:
        return path
    print("Descargando hand_landmarker.task (solo la primera vez)...")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, path)
    return path


def draw_feedback(frame: np.ndarray, expression: str, hand_gesture: str, active_meme: str) -> None:
    labels = {
        "surprise": ("Expression: SURPRISE", (0, 200, 255)),
        "cry": ("Expression: CRY", (255, 120, 120)),
        "pucker": ("Expression: PUCKER", (180, 105, 255)),
        "joy": ("Expression: JOY", (0, 255, 0)),
        "confused": ("Expression: CONFUSED", (200, 220, 100)),
        "neutral": ("Expression: NEUTRAL", (200, 200, 200)),
    }
    text, color = labels.get(expression, ("Expression: ?", (255, 255, 255)))
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    pred = "Prediction: 1" if expression != "neutral" else "Prediction: 0"
    pred_color = (0, 255, 0) if expression != "neutral" else (120, 120, 120)
    cv2.putText(frame, pred, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Hand: {hand_gesture}",
        (16, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (160, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Meme: {active_meme}",
        (16, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 140),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    meme_assets = load_expression_assets()
    hand_assets = load_optional_hand_assets()
    print("Imagenes cargadas:", ", ".join(EXPRESSION_IMAGES.values()))
    if hand_assets:
        print("Imagenes de mano cargadas:", ", ".join(HAND_GESTURE_IMAGES[k] for k in hand_assets))
    else:
        print("No se encontraron imagenes de mano dedicadas; usando fallback facial.")

    face_model_path = ensure_face_landmarker_model()
    hand_model_path = ensure_hand_landmarker_model()

    face_options = mp_face_landmarker.FaceLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(
            model_asset_path=str(face_model_path),
            delegate=mp_base_options.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    hand_options = mp_hand_landmarker.HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(
            model_asset_path=str(hand_model_path),
            delegate=mp_base_options.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    landmark_spec = mp_drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
    connection_spec = mp_drawing_utils.DrawingSpec(color=(90, 90, 90), thickness=1)
    smoother = ExpressionSmoother()
    hand_smoother = TemporalLabelSmoother(
        stable_frames=HAND_STABLE_FRAMES,
        initial="none",
        immediate_labels={"none"},
    )
    meme_smoother = TemporalLabelSmoother(stable_frames=MEME_STABLE_FRAMES, initial="neutral")

    try:
        face_landmarker_cm = mp_face_landmarker.FaceLandmarker.create_from_options(face_options)
        hand_landmarker_cm = mp_hand_landmarker.HandLandmarker.create_from_options(hand_options)
    except RuntimeError as exc:
        err = str(exc)
        if "kGpuService" in err or "NSOpenGL" in err or "OpenGL" in err:
            raise RuntimeError(
                "MediaPipe no pudo crear el contexto grafico (suele pasar sin ventana/pantalla, "
                "por ejemplo SSH o entornos sin GUI). Ejecuta la app en tu Mac desde Terminal.app "
                "o la terminal de Cursor con escritorio activo."
            ) from exc
        raise

    with face_landmarker_cm as face_landmarker, hand_landmarker_cm as hand_landmarker:
        frame_timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms += 33
            mp_image = MpImage(ImageFormat.SRGB, rgb)
            face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            expression = "neutral"
            hand_gesture = "none"

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                blend = blendshape_dict(face_result.face_blendshapes)
                mo, mw = landmark_scores(landmarks)
                expression = smoother.update(mo, mw, blend)

                mp_drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )

            detected_hand_gesture = "none"
            if hand_result.hand_landmarks:
                all_hands = hand_result.hand_landmarks
                if detect_heart_hands(all_hands):
                    detected_hand_gesture = "heart_hands"
                else:
                    first_hand = all_hands[0]
                    is_right_like = True
                    if hand_result.handedness and hand_result.handedness[0]:
                        handed = hand_result.handedness[0][0].category_name.lower()
                        is_right_like = handed == "right"
                    else:
                        wrist = first_hand[HAND_WRIST_ID]
                        is_right_like = wrist.x < 0.5
                    detected_hand_gesture = classify_hand_gesture(first_hand, is_right_like)

                for one_hand in all_hands:
                    mp_drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=one_hand,
                        connections=mp_hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_utils.DrawingSpec(
                            color=(0, 255, 180), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=2
                        ),
                    )

            hand_gesture = hand_smoother.update(detected_hand_gesture)
            active_meme = meme_smoother.update(resolve_trigger(expression, hand_gesture))
            meme = pick_meme_asset(meme_assets, hand_assets, active_meme, expression, hand_gesture)
            frame = overlay_meme(frame, meme)
            if SHOW_TRIGGER_WINDOW:
                trigger_view = meme[:, :, :3]
                cv2.imshow("Hampter Trigger", trigger_view)

            draw_feedback(frame, expression, hand_gesture, active_meme)
            cv2.imshow("Hampter Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    if SHOW_TRIGGER_WINDOW:
        cv2.destroyWindow("Hampter Trigger")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
