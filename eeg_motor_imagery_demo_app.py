import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="EEG Motor Imagery Demo", layout="wide")

st.title("EEG Motor Imagery Classification Demo")
st.write(
    "This lightweight demo app visualizes one EEG trial and runs a trained EEGNet model "
    "to predict left-hand or right-hand motor imagery."
)

with st.sidebar:
    st.header("App Setup")
    uploaded_model = st.file_uploader(
        "Upload trained EEGNet model (.h5 or .keras)",
        type=["h5", "keras"]
    )
    uploaded_trial = st.file_uploader(
        "Upload one EEG trial (.npy)",
        type=["npy"],
        help="Expected shape: (22, 1001) for one EEG trial."
    )
    uploaded_label = st.selectbox(
        "Optional true label",
        options=["Unknown", "Left hand", "Right hand"],
        index=0
    )


def load_npy_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    data = np.load(io.BytesIO(uploaded_file.read()), allow_pickle=False)
    return data


def normalize_trial(trial: np.ndarray) -> np.ndarray:
    mean = trial.mean()
    std = trial.std()
    return (trial - mean) / (std + 1e-8)


def prepare_for_eegnet(trial: np.ndarray) -> np.ndarray:
    """
    Input shape: (22, 1001)
    Output shape: (1, 22, 1001, 1)
    """
    trial = normalize_trial(trial)
    return trial[np.newaxis, ..., np.newaxis]


def plot_trial_lines(trial: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 6))
    offset = 8
    for i in range(trial.shape[0]):
        ax.plot(trial[i] + i * offset)
    ax.set_title("EEG Trial Across Channels")
    ax.set_xlabel("Time Points")
    ax.set_ylabel("Amplitude (offset for display)")
    return fig


def plot_heatmap(trial: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(trial, aspect="auto", cmap="viridis")
    ax.set_title("EEG Trial Heatmap")
    ax.set_xlabel("Time Points")
    ax.set_ylabel("Channels")
    fig.colorbar(im, ax=ax)
    return fig


trial = load_npy_from_upload(uploaded_trial)

if trial is not None:
    st.subheader("Trial Summary")
    st.write(f"Uploaded trial shape: {trial.shape}")

    if trial.shape != (22, 1001):
        st.error("The uploaded trial must have shape (22, 1001).")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_trial_lines(trial))

        with col2:
            st.pyplot(plot_heatmap(trial))

        if uploaded_label != "Unknown":
            st.info(f"True label: {uploaded_label}")

        st.subheader("Prediction")

        if uploaded_model is None:
            st.warning("Upload a trained EEGNet model to run prediction.")
        elif not TF_AVAILABLE:
            st.error("TensorFlow is not available in this environment.")
        else:
            try:
                model_path = uploaded_model.name

                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())

                model = tf.keras.models.load_model(model_path)
                x_input = prepare_for_eegnet(trial)
                probs = model.predict(x_input, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                pred_label = "Left hand" if pred_idx == 0 else "Right hand"

                st.success(f"Predicted class: {pred_label}")
                st.write(
                    {
                        "Left hand probability": float(probs[0]),
                        "Right hand probability": float(probs[1]),
                    }
                )

            except Exception as e:
                st.error(f"Model inference failed: {e}")
else:
    st.info(
        "Upload one EEG trial in .npy format to begin. "
        "You can export a single trial from your notebook using: "
        "np.save('sample_trial.npy', X[0][:22, :])"
    )

st.divider()

st.subheader("How to Export a Sample Trial from Your Notebook")
st.code(
    """# Save one EEG trial with 22 EEG channels
sample_trial = X[0][:22, :]
np.save("sample_trial.npy", sample_trial)
""",
    language="python"
)

st.subheader("How to Run the App")
st.code(
    """pip install streamlit tensorflow matplotlib numpy
streamlit run eeg_motor_imagery_demo_app.py
""",
    language="bash"
)