import os
import numpy as np
import math

# Add paths to DLLs (required if using OpenCV and CUDA with DLL dependencies on Windows)
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.1\bin\11.8")  # For cuDNN DLLs
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")  # For CUDA DLLs
os.add_dll_directory(r"C:\opencv\install\x64\vc16\bin")  # For OpenCV DLLs

import cv2

def translate_frame(frame, x_shift, y_shift):
    """
    Translates the given frame along the x and y axes by the specified shifts.
    """
    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # Apply the translation
    translated_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    return translated_frame

def process_video(input_path, output_path, frequency=0.05, amplitude_x=20, amplitude_y=15):
    """
    Reads a video, applies an elliptical translation to each frame, and writes the result to a new video.

    Args:
    - input_path (str): Path to the input video.
    - output_path (str): Path to save the output video.
    - frequency (float): Frequency of the sine and cosine wave for translation.
    - amplitude_x (int): Amplitude of the translation along the x-axis.
    - amplitude_y (int): Amplitude of the translation along the y-axis.
    """
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate translation amounts using sine and cosine waves
        x_shift = int(amplitude_x * math.cos(frequency * i))
        y_shift = int(amplitude_y * math.sin(frequency * i))
        
        # Apply translation to the frame
        translated_frame = translate_frame(frame, x_shift, y_shift)
        cv2.imshow("Translated Frame", translated_frame)
        cv2.waitKey(1)
        print(f"Processed frame {i+1}/{frame_count}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Write the translated frame to the output video
        out.write(translated_frame)

    print()

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete. Saved to:", output_path)

if __name__ == "__main__":
    # Example usage
    input_video_path = r"C:\Users\hadar\Documents\database\videos\other\9888614-hd_1920_1080_30fps.mp4"
    output_video_path = r"C:\Users\hadar\Documents\database\videos\other\output_translated_video.mp4"

    process_video(input_video_path, output_video_path, frequency=0.3, amplitude_x=50, amplitude_y=20)

