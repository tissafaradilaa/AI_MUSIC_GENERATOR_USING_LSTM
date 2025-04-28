import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.saving import register_keras_serializable
import tensorflow as tf

# Load notes data from a pickle file
def load_notes(file_path):
    """
    Load notes data from a pickle file.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        print(f"Loaded notes data from '{file_path}'.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found. Ensure the file exists and the path is correct.")
    except Exception as e:
        raise RuntimeError(f"Error loading notes data: {e}")

@register_keras_serializable()
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Define RMSE metric for comparison
def calculate_rmse(actual_sequence, predicted_sequence, element_to_int):
    """
    Calculate RMSE between actual and predicted sequences.
    """
    try:
        actual_indices = [element_to_int.get(note, -1) for note in actual_sequence]
        predicted_indices = [element_to_int.get(note, -1) for note in predicted_sequence]
        valid_pairs = [(a, p) for a, p in zip(actual_indices, predicted_indices) if a != -1 and p != -1]

        if not valid_pairs:
            raise ValueError("No valid indices found.")
        
        actual_indices, predicted_indices = zip(*valid_pairs)
        rmse = np.sqrt(mean_squared_error(actual_indices, predicted_indices))
        return rmse
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return None

# Load notes data
def load_trained_model(model_path, custom_objects=None):
    try:
        custom_objects = custom_objects or {"root_mean_squared_error": root_mean_squared_error}
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Loaded model from '{model_path}'.")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{model_path}' not found. Ensure the model is trained and saved correctly.")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Load the trained model
def load_trained_model(model_path, custom_objects=None):
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Loaded model from '{model_path}'.")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{model_path}' not found. Ensure the model is trained and saved correctly.")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Generate new music
def generate_music(model, sequence_length, vocab_len, int_to_element, start_pattern, num_notes=200):
    """
    Generate music notes using a trained model.
    """
    pattern = start_pattern[:sequence_length]  # Ensure pattern length matches sequence_length
    generated_notes = []

    print("Generating music...")
    for note_index in range(num_notes):
        # One-hot encode the input pattern
        input_pattern = np.zeros((1, sequence_length, vocab_len), dtype=np.float32)
        for i, idx in enumerate(pattern):
            if 0 <= idx < vocab_len:
                input_pattern[0, i, idx] = 1.0

        # Predict the next note
        prediction = model.predict(input_pattern, verbose=0)
        idx = np.argmax(prediction)
        result = int_to_element.get(idx, None)

        if result is None:
            print(f"Error: Predicted index {idx} not found in mapping. Stopping generation.")
            break

        generated_notes.append(result)
        pattern = pattern[1:] + [idx]

    print(f"Music generation complete. Total generated notes: {len(generated_notes)}")
    return generated_notes

# Convert generated notes to a MIDI file
def save_to_midi(predictions, output_file):
    offset = 0
    final_notes = []

    for pattern in predictions:
        if '.' in pattern or pattern.isdigit():
            # Handle chords
            notes_in_chord = pattern.split('.')
            temp_notes = []
            for curr_note in notes_in_chord:
                try:
                    new_note = note.Note(int(curr_note))
                    new_note.storedInstrument = instrument.Piano()
                    temp_notes.append(new_note)
                except ValueError:
                    print(f"Invalid note {curr_note}. Skipping.")
            new_chord = chord.Chord(temp_notes)
            new_chord.offset = offset
            final_notes.append(new_chord)
        else:
            # Handle single notes
            try:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                final_notes.append(new_note)
            except Exception as e:
                print(f"Error creating note {pattern}: {e}")
        offset += 0.5

    try:
        midi_stream = stream.Stream(final_notes)
        midi_stream.write('midi', fp=output_file)
        print(f"Music saved as '{output_file}'")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")

# Main execution
if __name__ == "__main__":
    # File paths
    notes_file = "notes_data.pkl"
    model_file = "model.keras"
    output_midi_file = "output.mid"

    # Load notes data
    data = load_notes(notes_file)
    notes = data["notes"]
    pitch_names = data["pitch_names"]
    vocab_len = data["vocab_len"]

    print(f"Using vocab_len: {vocab_len}")

    # Create mappings
    int_to_element = dict(enumerate(pitch_names))
    element_to_int = {element: num for num, element in int_to_element.items()}

    # Load the model
    model = load_trained_model(model_file)
    sequence_length = model.input_shape[1]

    print(f"Model input shape: {model.input_shape}")
    print(f"Sequence length: {sequence_length}, Vocab length: {vocab_len}")

    # Prepare input data
    test_input = [
        [element_to_int.get(note, -1) for note in notes[i:i + sequence_length]]
        for i in range(len(notes) - sequence_length)
    ]
    test_input = [seq for seq in test_input if -1 not in seq]  # Remove invalid sequences

    # Randomly select a starting pattern
    if not test_input:
        raise ValueError("No valid test input sequences found.")
    start_index = np.random.randint(len(test_input))
    start_pattern = test_input[start_index]

    # Generate new music
    generated_sequence = generate_music(model, sequence_length, vocab_len, int_to_element, start_pattern)

    # Calculate RMSE
    actual_sequence = notes[start_index:start_index + len(generated_sequence)]
    rmse = calculate_rmse(actual_sequence, generated_sequence, element_to_int)
    if rmse is not None:
        print(f"RMSE between generated music and actual sequence: {rmse:.4f}")
    else:
        print("Unable to calculate RMSE due to invalid data.")

    # Save to MIDI
    save_to_midi(generated_sequence, output_midi_file)
