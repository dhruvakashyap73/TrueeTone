==============================================
         "TrueeTone" Navigation Guide
==============================================

Project Title: TrueeTone - Audio Authenticity Detection
Purpose: Detect whether an audio file is Human (Real) or AI-generated (Fake)

----------------------------------------------
🔧 CORE PROJECT FILES
----------------------------------------------

1. trueetone_app.py
   - Main Streamlit frontend app.
   - Handles audio upload/recording, preprocessing, predictions, mel spectrogram visualization, and PDF report generation.

2. cnn_training.py
   - CNN training script using MFCC features.
   - Includes preprocessing, model architecture, training, evaluation, and saving the model as .h5 file.

3. dense_training.py
   - Dense Neural Network training script using MFCC features.
   - Trains a fully connected model for binary classification and exports a .h5 file.

4. feature_extraction_wav2vec.py
   - Uses pretrained Wav2Vec2.0 model to extract audio features.
   - Applies entropy-based rule to classify audio as AI or Human without additional training.

----------------------------------------------
📂 DATASET STRUCTURE
----------------------------------------------

/Dataset/
├── Sampled Audio/
│   ├── REAL/     → Contains real human voice audio samples.
│   └── FAKE/     → Contains AI-generated voice samples.

├── Test Audio/
│   ├── Test Real/ → Test samples for real audio.
│   └── Test Fake/ → Test samples for fake audio.

----------------------------------------------
📂 TRAINING ASSETS
----------------------------------------------

/Training/
├── Models/
│   ├── pre_trained_cnn_model.h5   → Saved CNN model used in frontend.
│   └── pre_trained_dense_model.h5 → Saved DNN model used in frontend.

├── Images/
│   ├── logo.png                  → Logo used in PDF and app.
│   ├── bg.jpeg                   → Banner image for homepage.
│   ├── ModelAccCNN.png           → CNN model training accuracy graph.
│   └── ModelAccDNN.png           → DNN model training accuracy graph.

----------------------------------------------
📄 DOCUMENTATION FILES
----------------------------------------------

README.md
   - Complete documentation about the project: aim, objectives, model description, deployment, and usage.

TrueeTone_Project_Report.pdf
   - Full academic report explaining system design, training, results, screenshots, and references.
   - (Ensure this file is added to your GitHub repo root.)

----------------------------------------------
🗃️ TEMPORARY FILES (GENERATED AT RUNTIME)
----------------------------------------------

These files are generated during processing and report generation:

- /tmp/audio_waveform.png          → Waveform plot of uploaded/recorded audio.
- /tmp/mel_spectrogram.png         → Mel spectrogram image.
- /tmp/mfcc.png                    → MFCC feature plot.
- /tmp/<audio_file>.wav            → Converted WAV file for processing.

----------------------------------------------
▶️ RUNNING THE PROJECT LOCALLY OR IN COLAB
----------------------------------------------

1. Install all required dependencies:
   pip install streamlit librosa tensorflow torchaudio transformers scipy matplotlib scikit-learn resampy pyngrok audio-recorder-streamlit fpdf2

2. Run the Streamlit app:
   streamlit run trueetone_app.py

3. (Optional - For Colab users)
   - Use ngrok to expose the Streamlit app publicly:
     from pyngrok import ngrok
     ngrok.set_auth_token("YOUR_TOKEN")
     public_url = ngrok.connect(8501)
     print(public_url)

----------------------------------------------
🧠 RECOMMENDED FILE EXPLORATION ORDER
----------------------------------------------

1. README.md                   → Project overview
2. trueetone_app.py            → Full frontend logic
3. cnn_training.py             → CNN model training process
4. dense_training.py           → DNN model training process
5. feature_extraction_wav2vec.py → Wav2Vec classifier implementation
6. TrueeTone_Project_Report.pdf → Final project report (attach separately)

----------------------------------------------
📘 NEED MORE INFORMATION?
----------------------------------------------

For full methodology, results, and technical breakdown,  
refer to: TrueeTone_Project_Report.pdf

====================================================
