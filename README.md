import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import os
import csv

# === SETUP ===
video_path = "overwatch_clip.mp4"
model_path = "models/overwatch_heroes.tflite"  # You’ll need a custom-trained model
output_dir = "overwatch_outputs"
os.makedirs(output_dir, exist_ok=True)

# === LOAD TFLITE MODEL ===
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1:3]

# === VIDEO PROCESSING ===
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = 0
player_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, tuple(input_size))
    input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # If class IDs are supported
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame_players = []
    for i in range(len(scores)):
        if scores[i] > 0.6:
            ymin, xmin, ymax, xmax = boxes[i]
            x = int((xmin + xmax) / 2 * frame_width)
            y = int((ymin + ymax) / 2 * frame_height)

            hero_id = int(classes[i]) if i < len(classes) else -1
            label = f"Hero {hero_id}" if hero_id != -1 else "Player"
            frame_players.append((x, y))

            # Visualize hero label
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    player_positions.extend(frame_players)

    # Save frame every 45 frames (~1.5s at 30fps)
    if frame_count % 45 == 0:
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)

    frame_count += 1

cap.release()

# === HEATMAP: Zone Awareness (Map-based View) ===
heatmap = np.zeros((720, 1280))
for x, y in player_positions:
    if 0 <= x < 1280 and 0 <= y < 720:
        heatmap[y, x] += 1

plt.imshow(heatmap, cmap='inferno')
plt.title("Overwatch Zone Control Heatmap")
plt.axis('off')
plt.savefig(f"{output_dir}/heatmap_overwatch.png", bbox_inches='tight')

# === CSV EXPORT for Dashboards ===
with open(f"{output_dir}/player_positions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for x, y in player_positions:
        writer.writerow([x, y])

print("Overwatch e-sports analysis complete!")
