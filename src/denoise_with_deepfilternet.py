import os
import numpy as np
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

def denoise_with_deepfilternet(input_path, output_path, model=None, df_state=None):
    """
    Denoise audio using DeepFilterNet
    
    Args:
        input_path (str): Path to the noisy audio file
        output_path (str): Path to save the denoised audio
        model: Pre-loaded DeepFilterNet model (optional)
        df_state: Pre-loaded DeepFilterNet state (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize model if not provided
        if model is None or df_state is None:
            print("Loading DeepFilterNet model...")
            model, df_state, _ = init_df()
        
        # Load audio
        print(f"Loading audio from: {input_path}")
        audio, _ = load_audio(input_path, sr=df_state.sr())
        
        # Enhance/denoise the audio
        print("Denoising audio...")
        enhanced_audio = enhance(model, df_state, audio)
        
        # Save the enhanced audio
        print(f"Saving denoised audio to: {output_path}")
        # Use the model's sample rate directly
        sr_int = df_state.sr()
        save_audio(output_path, enhanced_audio, sr_int)
        
        print("Denoising completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during denoising: {str(e)}")
        return False

def batch_denoise(input_dir, output_dir, file_extension=".wav"):
    """
    Denoise multiple audio files in a directory
    
    Args:
        input_dir (str): Directory containing noisy audio files
        output_dir (str): Directory to save denoised audio files
        file_extension (str): Audio file extension to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once for batch processing
    print("Loading DeepFilterNet model for batch processing...")
    model, df_state, _ = init_df()
    
    # Process all audio files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(file_extension):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"denoised_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"\nProcessing: {filename}")
            denoise_with_deepfilternet(input_path, output_path, model, df_state)

if __name__ == "__main__":
    # Example usage
    
    # Single file denoising
    input_file = "/workspaces/Evaluating-Noise-Reduction-Techniques/tests/noisy/mixed_raw_1_20s.wav"
    output_file = "/workspaces/Evaluating-Noise-Reduction-Techniques/src/denoised_deepfilternet.wav"
    
    if os.path.exists(input_file):
        print("=== Single File Denoising ===")
        success = denoise_with_deepfilternet(input_file, output_file)
        if success:
            print(f"Denoised audio saved to: {output_file}")
    else:
        print(f"Input file not found: {input_file}")
    
    # Batch processing example (uncomment to use)
    # print("\n=== Batch Processing ===")
    # input_dir = "/workspaces/Evaluating-Noise-Reduction-Techniques/tests/noisy"
    # output_dir = "/workspaces/Evaluating-Noise-Reduction-Techniques/tests/denoised/deepfilternet"
    # batch_denoise(input_dir, output_dir)
    