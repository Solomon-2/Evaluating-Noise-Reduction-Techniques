import numpy as np
import librosa
import soundfile as sf
import argparse
import os

def insert_silence(
    input_path,
    output_path,
    silence_duration_s=10.0,
    insert_time_s=2.0
):
    """
    Loads an audio file, injects a period of silence, and saves the result.

    This function "injects" silence, meaning it splits the original audio
    at the insertion point and places the silence in between, making the
    total duration of the file longer.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the modified audio file.
        silence_duration_s (float): Duration of the silence to insert in seconds.
        insert_time_s (float): The timestamp in the original audio where the
                               silence will be inserted.

    Returns:
        tuple: A tuple (start_time, end_time) of the inserted silent segment,
               which is the ground truth for an apnea event. Returns None on error.
    """
    try:
        # 1. Load the original audio file
        y_original, sr = librosa.load(input_path, sr=None) # sr=None to preserve original sr
        original_duration = librosa.get_duration(y=y_original, sr=sr)
        print(f"Loaded '{input_path}' (Duration: {original_duration:.2f}s, SR: {sr}Hz)")

        # 2. Basic validation
        if insert_time_s > original_duration:
            print(f"Error: Insertion time ({insert_time_s:.2f}s) is after the end of the audio ({original_duration:.2f}s).")
            return None
        if silence_duration_s <= 0:
            print("Error: Silence duration must be positive.")
            return None

        # 3. Create the silent segment
        num_silence_samples = int(silence_duration_s * sr)
        silence_array = np.zeros(num_silence_samples, dtype=np.float32)

        # 4. Find the split point in the original audio
        split_point_samples = int(insert_time_s * sr)
        
        # 5. Split the original audio and concatenate with silence
        part1 = y_original[:split_point_samples]
        part2 = y_original[split_point_samples:]
        y_modified = np.concatenate([part1, silence_array, part2])

        # 6. Save the modified audio
        sf.write(output_path, y_modified, sr)
        print(f"Successfully saved modified audio to '{output_path}'")
        
        # 7. Define the ground truth
        # The apnea event starts at the insertion time and lasts for the silence duration
        ground_truth_event = (insert_time_s, insert_time_s + silence_duration_s)
        
        return ground_truth_event

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# This makes the script runnable from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Injects a period of silence into an audio file to simulate an apnea event.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input audio file.")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path for the output audio file.\n(Default: adds '_with_apnea' to the input file name)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of the silence to insert (in seconds).\n(Default: 10.0)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=2.0,
        help="Time in the original audio to insert the silence (in seconds).\n(Default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Generate a default output file name if none is provided
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_with_apnea{ext}"

    print("-" * 30)
    
    # Run the main function
    ground_truth = insert_silence(
        input_path=args.input_file,
        output_path=args.output_file,
        silence_duration_s=args.duration,
        insert_time_s=args.time
    )

    if ground_truth:
        print("-" * 30)
        print("Simulation Summary:")
        print(f"  GROUND TRUTH Apnea Event Start: {ground_truth[0]:.2f} seconds")
        print(f"  GROUND TRUTH Apnea Event End:   {ground_truth[1]:.2f} seconds")
        print("-" * 30)
        print("\nThis new file is now ready for your testing pipeline!")