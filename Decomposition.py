import cv2, os

video_path = "Eupen_1DOF_track01_240fps.mp4"
output_directory = "ToCrop"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

video_capture = cv2.VideoCapture(video_path)

frame_count = 0

while frame_count <= 5:
    success, frame = video_capture.read()

    if not success:
        break

    output_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_path, frame)

    frame_count += 1

video_capture.release()
print(f"Total frames extracted: {frame_count}")
