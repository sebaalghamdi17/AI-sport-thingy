import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import time
import os

# --- Initialization ---
video_path = "game_footage.mp4"
output_dir = "media_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="ssd_mobilenet_v2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start video processing
cap = cv2.VideoCapture(video_path)
frame_count = 0
player_tracks = []

# --- Process Video Frame by Frame ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_size = input_details[0]['shape'][1:3]
    resized = cv2.resize(frame, tuple(input_size))
    input_data = np.expand_dims(resized.astype(np.uint8), axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Track players
    frame_players = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            x_center = int((xmin + xmax) / 2 * frame.shape[1])
            y_center = int((ymin + ymax) / 2 * frame.shape[0])
            frame_players.append((x_center, y_center))
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

    player_tracks.extend(frame_players)

    # Optional: Save annotated frame
    if frame_count % 30 == 0:
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)

    frame_count += 1

cap.release()

# --- Generate Heatmap for Media Use ---
heatmap_img = np.zeros((720, 1280), dtype=np.float32)

for (x, y) in player_tracks:
    if 0 <= x < 1280 and 0 <= y < 720:
        heatmap_img[y, x] += 1

plt.imshow(heatmap_img, cmap='hot', interpolation='nearest')
plt.title("Player Activity Heatmap")
plt.axis('off')
plt.savefig(f"{output_dir}/heatmap_media.png", bbox_inches='tight')

print("Media outputs generated!")
