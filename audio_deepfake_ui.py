import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fpdf import FPDF
from pathlib import Path
import tempfile

PROJECT_DIR = Path(__file__).resolve().parent

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css(PROJECT_DIR / "style_all.css")

# Load model
model = tf.keras.models.load_model(PROJECT_DIR / "cnn_model_smaller.keras")

st.title("🔍 Audio Deepfake Detection (Forensic Tool)")
st.write("Upload an audio file (wav/mp3/ogg) and the system will analyze it for authenticity.")

# Upload audio
audio_file = st.file_uploader("Upload audio file (wav/mp3/ogg)", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    file_extension = Path(audio_file.name).suffix.lower()
    audio_format = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
    }[file_extension]
    st.audio(audio_bytes, format=audio_format)

    with st.spinner("Processing audio and generating spectrogram..."):
        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, duration=5)

        # Generate spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save spectrogram
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='magma')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        st.image(buf, caption="Generated Spectrogram", use_container_width=True)

        # Prepare image for model
        img = Image.open(buf).convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        st.write(f"Raw model output: {pred}")  #  See raw output for debug

        # Determine logic based on output shape
        if pred.shape[1] == 1:
            # Sigmoid case
            prob_fake = pred[0][0]
            label = 'FAKE' if prob_fake > 0.5 else 'REAL'
            confidence = prob_fake if label == 'FAKE' else 1 - prob_fake
        else:
            # Softmax case
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            label = 'REAL' if class_idx == 0 else 'FAKE'

        st.success(f"Prediction: **{label}** (confidence: {confidence:.2f})")

        # Export PDF
        if st.button("Export Result as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, 'Audio Deepfake Detection Report', ln=True, align='C')
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f'Prediction: {label}', ln=True)
            pdf.cell(0, 10, f'Confidence: {confidence:.2f}', ln=True)

            # Save image
            temp_img = "temp_spec.png"
            with open(temp_img, "wb") as f:
                f.write(buf.getvalue())
            pdf.image(temp_img, w=150)

            # Save and download
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = Path(temp_dir) / "deepfake_report.pdf"
                pdf.output(str(pdf_path))
                pdf_bytes = pdf_path.read_bytes()

            st.download_button("Download PDF Report", data=pdf_bytes, file_name="deepfake_report.pdf", mime="application/pdf")
