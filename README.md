# AI Music Generator using LSTM

Overview
This project is an AI music generator that uses a Deep Learning model based on Long Short-Term Memory (LSTM) networks to generate piano music. The model is trained on a dataset of MIDI files containing various piano compositions, which it uses to learn patterns and structures in music. Once trained, the model can generate new musical sequences in the style of the training data.

Features
1. MIDI Parsing: Extracts notes and chords from MIDI files using the music21 library.
2. LSTM Model: Uses an LSTM-based neural network to learn and predict musical patterns.
3. Music Generation: Generates new music sequences based on the learned patterns.
4. Customizable Training: Allows adjustment of parameters like sequence length, batch size, and epochs for training.

Requirements
To run this project, you need the following libraries installed:
1. Python 3.x
2. music21
3. TensorFlow/Keras
4. NumPy
5. glob
6. pickle

How It Works
1. Data Preprocessing:
   - MIDI files are parsed to extract notes and chords.
   - The extracted data is converted into numerical sequences for training.
   - Sequences are created to serve as input and target pairs for the LSTM model.

2. Model Training:
   - An LSTM-based neural network is trained on the preprocessed sequences.
   - The model learns to predict the next note or chord in a sequence based on the previous notes.

3. Music Generation:
   - The trained model generates new music by predicting sequences of notes.
   - The generated sequences are converted back into MIDI format for playback or further processing.
  
Output
The model generates MIDI files that can be played using any MIDI-compatible software or converted to audio formats for listening.
