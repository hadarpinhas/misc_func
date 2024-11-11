import os
# Add paths to DLLs (required if using OpenCV and CUDA with DLL dependencies on Windows)
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.1\bin\11.8")  # For cuDNN DLLs
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")  # For CUDA DLLs
os.add_dll_directory(r"C:\opencv\install\x64\vc16\bin")  # For OpenCV DLLs
import cv2
from ultralytics import YOLO, solutions

# Define paths:
path_input_video = r"C:\Users\hadar\Documents\database\videos\other\9888614-hd_1920_1080_30fps.mp4"
path_output_video = r"C:\Users\hadar\Documents\database\videos\other\output_translated_video.mp4"
path_model = "/path/to/your/yolo_model.pt"

# Initialize YOLOv8 Detection Model
model = YOLO(path_model)

# Initialize Object Counter
counter = solutions.ObjectCounter( 
  view_img=True,                     # Display the image during processing 
  reg_pts=[(512, 320), (512, 1850)], # Region of interest points 
  classes_names=model.names,         # Class names from the YOLO model 
  draw_tracks=True,                  # Draw tracking lines for objects 
  line_thickness=2,                  # Thickness of the lines drawn 
  )

# Open the Video File
cap = cv2.VideoCapture(path_input_video) 
assert cap.isOpened(), "Error reading video file"

# Initialize the Video Writer to save resulted video
video_writer = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1080, 1920))

# itterate over video frames:
frame_count = 0 
while cap.isOpened(): 
  success, frame = cap.read() 
  if not success: 
    print("Video frame is empty or video processing has been successfully completed.") 
    break 

  # Perform object tracking on the current frame 
  tracks = model.track(frame, persist=True, tracker='botsort.yaml', iou=0.2) 

  # Use the Object Counter to count objects in the frame and get the annotated image 
  frame = counter.start_counting(frame, tracks) 

  # Write the annotated frame to the output video 
  video_writer.write(frame) 
  frame_count += 1

# Release all Resources:
cap.release() 
video_writer.release() 
cv2.destroyAllWindows()

# Print counting results:
print(f'In: {counter.in_counts}\nOut: {counter.out_counts}\nTotal: {counter.in_counts + counter.out_counts}')
print(f'Saves output video to {path_output_video}')