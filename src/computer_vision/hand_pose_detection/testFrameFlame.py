import cv2
cap = cv2.VideoCapture("bow too high-slow (3).mp4")
ret, frame = cap.read()
frame = cv2.flip(frame, 0)  # fix upside-down decode on Linux
cv2.imwrite("test_frame_1.jpg", frame)
cap.release()
