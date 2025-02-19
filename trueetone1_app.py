import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import torch
import torchaudio
import io
import os
from PIL import Image
from scipy.stats import skew, kurtosis, median_abs_deviation
import torch.nn.functional as F  # Alias the functional module as F
import matplotlib.pyplot as plt  # Import matplotlib explicitly
from io import BytesIO
from pydub import AudioSegment

# Function to load image
def load_image(image_path):
    try:
        with open(image_path, "rb") as file:
            image_data = file.read()
            if not image_data:
                st.error(f"Image file {image_path} is empty.")
                return None
            return Image.open(io.BytesIO(image_data))
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Load Models (Ensure these paths are correct for your Google Drive)
@st.cache_resource()
def load_models():
    try:
        dnn_model = tf.keras.models.load_model("/content/drive/MyDrive/TrueeTone/Training/pre_trained_dense_model.h5")
        cnn_model = tf.keras.models.load_model("/content/drive/MyDrive/TrueeTone/Training/pre_trained_cnn_model.h5")  # Change path if needed
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        wav2vec_model = bundle.get_model()
        return dnn_model, cnn_model, wav2vec_model, bundle
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

dnn_model, cnn_model, wav2vec_model, bundle = load_models()

def convert_to_wav(audio_file):
    """Converts an uploaded audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(audio_file)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        st.error(f"Error converting audio to WAV: {e}")
        return None

# Prediction Functions
def predict_dnn(audio_file_content, file_name):
    try:
        # Convert to WAV
        wav_audio = convert_to_wav(io.BytesIO(audio_file_content))
        if wav_audio is None:
            return None

        # Save converted WAV file to a temporary location
        temp_file_path = f"/tmp/{file_name}.wav"  # Append .wav extension
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(wav_audio.read())

        sound_signal, sample_rate = librosa.load(temp_file_path, res_type="kaiser_fast")
        mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
        mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
        mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
        result_array = dnn_model.predict(mfccs_features_scaled)
        result_classes = ["FAKE", "REAL"]
        result = np.argmax(result_array[0])
        return result_classes[result]
    except Exception as e:
        st.error(f"Error in DNN prediction: {e}")
        return None

def predict_cnn(audio_file_content, file_name):
    try:
        # Convert to WAV
        wav_audio = convert_to_wav(io.BytesIO(audio_file_content))
        if wav_audio is None:
            return None

        # Save converted WAV file to a temporary location
        temp_file_path = f"/tmp/{file_name}.wav"  # Append .wav extension
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(wav_audio.read())

        sound_signal, sample_rate = librosa.load(temp_file_path, res_type="kaiser_fast")
        mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
        mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
        mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
        result_array = cnn_model.predict(mfccs_features_scaled)
        result_classes = ["FAKE", "REAL"]
        result = np.argmax(result_array[0])
        return result_classes[result]
    except Exception as e:
        st.error(f"Error in CNN prediction: {e}")
        return None

def extract_features(audio_file_content, file_name, bundle, model):
    try:
        # Convert to WAV
        wav_audio = convert_to_wav(io.BytesIO(audio_file_content))
        if wav_audio is None:
            return None

        # Save converted WAV file to a temporary location
        temp_file_path = f"/tmp/{file_name}.wav"  # Append .wav extension
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(wav_audio.read())

        waveform, sample_rate = torchaudio.load(temp_file_path)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)(waveform)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform)

        pooled_features = []
        for f in features:
            if f.dim() == 3:
                f = f.permute(0, 2, 1)
                pooled_f = F.adaptive_avg_pool1d(f[0].unsqueeze(0), 1).squeeze(0)
                pooled_features.append(pooled_f)

        final_features = torch.cat(pooled_features, dim=0).numpy()
        final_features = (final_features - np.mean(final_features)) / (np.std(final_features) + 1e-10)
        return final_features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def additional_features(features):
    if features is None:
        return None, None
    mad = median_abs_deviation(features)
    features_clipped = np.clip(features, 1e-10, None)
    entropy = -np.sum(features_clipped * np.log(features_clipped))
    return mad, entropy

def classify_audio(features):
    if features is None:
        return None, None
    mean_value = np.mean(features)
    variance_value = np.var(features)
    skewness_value = skew(features)[0]
    kurtosis_value = kurtosis(features)[0]
    _, entropy = additional_features(features)
    if entropy is None:
        return None, None
    if entropy > 200:
        return "Human", entropy
    else:
        return "AI", entropy

def predict_wav2vec(audio_file_content, file_name, bundle, model):
    try:
        features = extract_features(audio_file_content, file_name, bundle, model)
        if features is not None:
            prediction, entropy = classify_audio(features)
            return prediction, entropy
        else:
            return None, None
    except Exception as e:
        st.error(f"Error in Wav2Vec prediction: {e}")
        return None, None

# Spectrogram Function
def plot_mel_spectrogram(audio_file_content, file_name):
    try:
        # Convert to WAV for plotting spectrogram.
        wav_audio = convert_to_wav(io.BytesIO(audio_file_content))
        if wav_audio is None:
            return

        # Save converted WAV file to a temporary location
        temp_file_path = f"/tmp/{file_name}.wav"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(wav_audio.read())

        y, sr = librosa.load(temp_file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error plotting spectrogram: {e}")

# Main Streamlit App
def main():
    # Load and display the background image
    bg_image = load_image('bg.png')
    if bg_image:
        st.image(bg_image)
    st.title("TrueeTone: Audio Authenticity Detection")

    # Audio Input
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

    if audio_file is not None:
        file_extension = os.path.splitext(audio_file.name)[1].lower()

        if file_extension not in ['.wav', '.mp3', '.ogg']:
            st.error("Unsupported file format. Please upload a WAV, MP3, or OGG file.")
            return

        file_name = audio_file.name

        # Read the content of the audio file
        audio_file_content = audio_file.read()

        # Convert to WAV for audio display and processing
        wav_audio = convert_to_wav(io.BytesIO(audio_file_content))

        if wav_audio is None:
            return

        st.audio(audio_file_content, format=f'audio/{file_extension[1:]}')

        # Display Mel Spectrogram
        st.subheader("Mel Spectrogram")
        plot_mel_spectrogram(audio_file_content, file_name)

        # Predictions
        st.subheader("Predictions")

        dnn_prediction = predict_dnn(audio_file_content, file_name)

        cnn_prediction = predict_cnn(audio_file_content, file_name)

        wav2vec_prediction, entropy = predict_wav2vec(audio_file_content, file_name, bundle, wav2vec_model)

        if dnn_prediction:
            st.write(f"DNN Prediction: {dnn_prediction}")

        if cnn_prediction:
            st.write(f"CNN Prediction: {cnn_prediction}")

        if wav2vec_prediction:
            st.write(f"Wav2Vec Prediction: {wav2vec_prediction} (Entropy: {entropy:.2f})")

        # Best Prediction (Based on Wav2Vec)
        st.subheader("Best Prediction (Wav2Vec)")
        if wav2vec_prediction == "AI":
            st.warning("This audio is likely AI-generated.")
        elif wav2vec_prediction == "Human":
            st.success("This audio is likely Human-generated.")
        else:
            st.write("Unable to determine audio authenticity.")

if __name__ == "__main__":
    main()
