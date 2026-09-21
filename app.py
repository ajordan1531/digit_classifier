import joblib
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------- config ----
MODEL_PATH = "digit_model.joblib"

# Pixel value that means "fully white" in YOUR training data:
#   255.0 -> trained on raw MNIST pixels (0-255)
#   1.0   -> trained on MNIST divided by 255
#   16.0  -> trained on sklearn's built-in load_digits (8x8, values 0-16)
# Leave as None to guess from the model's input size (64 features -> 16, else 255).
PIXEL_MAX = 1.0
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Digit Classifier", page_icon="✏️")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

n_features = int(model.n_features_in_)
SIDE = int(round(n_features ** 0.5))  # 28 for 784 features, 8 for 64 features
assert SIDE * SIDE == n_features, f"Expected a square image, got {n_features} features"
pixel_max = PIXEL_MAX if PIXEL_MAX is not None else (16.0 if SIDE == 8 else 255.0)


def preprocess(rgba: np.ndarray):
    """Canvas image -> flat feature vector shaped like the training data."""
    gray = Image.fromarray(rgba.astype("uint8")).convert("L")
    arr = np.array(gray)

    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return None  # nothing drawn yet

    # Crop to the digit, shrink it to fit inside the frame with a margin,
    # then center it (MNIST digits are laid out like this: 20px box in 28px frame).
    box = max(1, round(SIDE * 20 / 28))
    digit = gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = box / max(digit.size)
    new_size = (max(1, round(digit.width * scale)), max(1, round(digit.height * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    frame = Image.new("L", (SIDE, SIDE), 0)
    frame.paste(digit, ((SIDE - digit.width) // 2, (SIDE - digit.height) // 2))
    return np.array(frame, dtype="float32") / 255.0 * pixel_max


st.title("✏️ Handwritten Digit Classifier")
st.write("Draw a single digit (0–9) in the box and the model will guess what it is.")

col_draw, col_result = st.columns(2)

with col_draw:
    canvas = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,  # required in streamlit-drawable-canvas 0.10+
        key="canvas",
    )

with col_result:
    if canvas.image_data is not None:
        img = preprocess(canvas.image_data)
        if img is not None:
            x = img.reshape(1, -1)  # sklearn wants (n_samples, n_features)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                pred = model.classes_[int(np.argmax(probs))]
                st.metric("Prediction", int(pred))
                st.caption(f"Confidence: {probs.max():.1%}")
                st.bar_chart(dict(zip(map(str, model.classes_), probs)))
            else:
                # e.g. LinearSVC or SVC(probability=False): no probabilities available
                st.metric("Prediction", int(model.predict(x)[0]))

            with st.expander(f"What the model sees ({SIDE}×{SIDE})"):
                st.image(img / pixel_max, width=140, clamp=True)
        else:
            st.info("Waiting for a drawing…")
