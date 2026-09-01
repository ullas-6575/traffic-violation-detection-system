import cv2

url = "http://192.168.10.191:8080/video"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

print("Opened:", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    cv2.imshow("Android Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()