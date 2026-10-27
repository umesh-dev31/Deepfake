# Audio Deepfake Detection

A Streamlit application that analyzes an audio clip by converting it to a
mel-spectrogram and classifying the spectrogram with a convolutional neural
network. The app can also export the prediction and spectrogram as a PDF report.

This project is an experimental classifier, not a forensic guarantee. Treat
predictions as indicative and validate the model on representative data before
using it for consequential decisions.

## Features

- Upload WAV, MP3, or OGG audio in the Streamlit app.
- Generate a mel-spectrogram from up to the first five seconds of audio.
- Classify the spectrogram as real or fake with the saved CNN model.
- Export a PDF report containing the result and spectrogram.
- Convert WAV datasets to spectrogram images for model training.
- Generate SHAP explanations and run multilingual evaluation scripts.

## Project files

| File | Purpose |
| --- | --- |
| `audio_deepfake_ui.py` | Streamlit app for audio upload, prediction, and PDF export |
| `image_ui.py` | Streamlit app for classifying uploaded spectrogram images |
| `preprocess_spectrograms.py` | Converts WAV datasets into mel-spectrogram images |
| `train_cnn_spectrogram.py` | Trains and evaluates the CNN |
| `test_multilang.py` | Evaluates the model on multilingual spectrograms |
| `real_time_predict.py` | Runs a prediction on a configured spectrogram image |
| `explain_shap.py` | Generates SHAP explanations for a batch of spectrograms |
| `cnn_model_smaller.keras` | Pre-trained model used by the Streamlit apps |
| `style_all.css` | Streamlit app styling |

## Requirements

- Python 3.10 or later
- The project model file, `cnn_model_smaller.keras`
- The Python packages below

Create and activate a virtual environment from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit librosa matplotlib numpy pillow tensorflow fpdf2 scikit-learn shap
```

If PowerShell prevents activation, run commands using the virtual environment's
interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit librosa matplotlib numpy pillow tensorflow fpdf2 scikit-learn shap
```

Some OGG/Opus files may require a compatible audio decoder in addition to the
Python packages. If an OGG upload is accepted but cannot be decoded, install
FFmpeg and ensure `ffmpeg` is available on `PATH`.

## Run the audio app

From the project directory:

```powershell
streamlit run audio_deepfake_ui.py
```

Open the local URL printed by Streamlit, upload a WAV, MP3, or OGG file, and
wait for the prediction. Use **Export Result as PDF** to prepare a report, then
select **Download PDF Report**.

The app loads `style_all.css` and `cnn_model_smaller.keras` from the same
directory as `audio_deepfake_ui.py`. Keep both files there or update the paths
in the script.

To run the spectrogram-image app instead:

```powershell
streamlit run image_ui.py
```

## Dataset preparation and training

The preprocessing script expects English WAV files arranged as:

```text
dataset/
  english/
    real/
    fake/
  lang/
    language_name/
      real/
      fake/
```

Before running the data or model scripts, update the dataset, output, and model
paths near the top of each script. Several scripts currently contain
machine-specific `G:\final Audiodeepfake\...` paths and will not work on another
computer until those paths are changed.

Run the workflow in order:

1. Put the WAV datasets in the expected folder structure and configure paths
   in `preprocess_spectrograms.py`.
2. Run `python preprocess_spectrograms.py` to generate spectrogram images.
3. Configure the processed-data and output-model paths in
   `train_cnn_spectrogram.py`.
4. Run `python train_cnn_spectrogram.py` to train the CNN and save the model.
5. Configure the model and multilingual-data paths in `test_multilang.py` to
   evaluate multilingual samples.

The training script uses a binary sigmoid output and saves the trained model
to the configured path. For predictions from the Streamlit app to use that
model, save or copy it to `cnn_model_smaller.keras` in the project directory.

## Notes

- The preprocessing script currently processes WAV files only.
- The Streamlit audio app analyzes at most the first five seconds of an upload.
- Dataset files are not included in this repository; provide your own data and
  follow its licensing and privacy requirements.
- Do not use the reported prediction confidence as a calibrated probability
  unless calibration has been tested separately.
