import cv2

# Initialize webcam
video = cv2.VideoCapture(0)  # Use 0 for the default camera
if not video.isOpened():
    raise RuntimeError("Could not open webcam. Please check your camera connection.")

# Display webcam feed
try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # Display the frame in a window
        cv2.imshow('Webcam Feed', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    video.release()
    cv2.destroyAllWindows()
