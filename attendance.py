import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and their encodings
eamans_face = face_recognition.load_image_file("D:/python project/Attendance system/Faces/WIN_20240301_01_02_16_Pro.jpg")
eaman_encoding = face_recognition.face_encodings(eamans_face)[0]
e_face = face_recognition.load_image_file("D:/python project/Attendance system/Faces/WhatsApp Image 2024-03-08 at 16.53.14_ead54ba5.jpg")
e_encoding = face_recognition.face_encodings(e_face)[0]

# Define known faces and their corresponding names
known_faces = [eaman_encoding, e_encoding]
known_faces_names = ["eaman", "mayank"]

# Initialize list of students (names)
students = known_faces_names.copy()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Get current date
current_date = datetime.now().strftime("%y-%m-%d")

# Open CSV file in append mode
with open(f"{current_date}.csv", "a", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Main loop for capturing video frames
    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Iterate through detected faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distance = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distance)
            
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (face_location[3], face_location[0])
                font_scale = 1
                font_color = (255, 0, 0)
                thickness = 2
                line_type = 2
                cv2.putText(frame, name + " Present", bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
                
                # If student is detected, remove from list and write to CSV
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    csv_writer.writerow([name, current_time])

        # Display frame with attendance status
        cv2.imshow("Attendance", frame)
        
        # Check for user input to exit loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
