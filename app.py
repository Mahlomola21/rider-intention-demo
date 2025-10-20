import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# === Load trained model ===
MODEL_PATH = Path("output/pooled_model.joblib")
try:
    model_obj = joblib.load(MODEL_PATH)
    model = model_obj["model"]
    scaler = model_obj.get("scaler", None)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Define label mapping ---
CLASS_ORDER = [
    "Left Lane Change",  # 0
    "Left Turn",  # 1
    "Right Lane Change",  # 2
    "Right Turn",  # 3
    "Slow-Stop",  # 4
    "Straight",  # 5
]

st.title("Two-Wheeler Rider Intention Prediction")
st.write(
    """
Upload a `.npy` file containing VGG16 features (pooled or raw). 
The app will preprocess and predict the rider's intention.
"""
)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a .npy file", type="npy")

if uploaded_file is not None:
    try:
        arr = np.load(uploaded_file, allow_pickle=True)
        st.write(f"Original feature shape: {arr.shape}")

        # --- Validate shape ---
        if arr.size == 0:
            st.warning("⚠️ Uploaded file is empty.")
        else:
            # --- Automatic pooling if multiple frames ---
            if arr.ndim > 1:
                try:
                    arr = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
                except Exception:
                    st.warning(
                        "⚠️ Could not compute mean/std, using flattened array instead."
                    )
                    arr = arr.flatten()
            else:
                arr = arr.flatten()

            # --- Ensure shape matches model ---
            if scaler is not None:
                arr = arr.reshape(1, -1)
                if arr.shape[1] != scaler.mean_.shape[0]:
                    st.warning(
                        f"⚠️ Feature size mismatch. Model expects {scaler.mean_.shape[0]} features."
                    )
                arr = scaler.transform(arr)
            else:
                arr = arr.reshape(1, -1)

            # --- Predict ---
            try:
                pred_num = model.predict(arr)[0]
                if pred_num < 0 or pred_num >= len(CLASS_ORDER):
                    st.warning(f"⚠️ Predicted class {pred_num} is out of range.")
                    pred_label = "Unknown"
                else:
                    pred_label = CLASS_ORDER[pred_num]

                st.success(f"Predicted intention: **{pred_label}**")

                # --- Probabilities ---
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(arr)[0]
                    st.subheader("Prediction probabilities:")
                    for i, cls_name in enumerate(CLASS_ORDER):
                        st.write(f"{cls_name}: {proba[i]:.3f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Failed to load or process the uploaded file: {e}")
