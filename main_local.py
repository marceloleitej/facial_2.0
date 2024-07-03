import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime, timedelta
import json

# Load student data from JSON file
with open('studentsData.json', 'r') as f:
    studentsData = json.load(f)["Students"]

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the background image
imgBackground = cv2.imread('Resources/background.png')

# Import mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the known face encodings and student IDs from the pickle file
print("Load Encode File ....")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Initialize variables
modeType = 0
counter = 0
id = -1
imgStudent = []
display_start_time = None  # Variable to store the start time of data display
current_displayed_id = None  # Variable to store the currently displayed student ID

# Dictionary to store the time of last recognition for each student
recognized_students = {}

def get_student_data(id):
    """
    Function to get student information and image.
    """
    studentInfo = studentsData[str(id)]
    imgStudentPath = f'Images/{id}.png'
    imgStudent = cv2.imread(imgStudentPath)
    return studentInfo, imgStudent

frame_count = 0
process_frame_rate = 5  # Process every 5 frames
ignore_duration = timedelta(seconds=300)  # Duration to ignore already recognized faces
display_duration = 4  # Duration to display student data

while True:
    success, img = cap.read()
    frame_count += 1

    if frame_count % process_frame_rate == 0:
        print(f"Frame Count: {frame_count}, Counter: {counter}, ModeType: {modeType}")

        # Resize and convert the image for face recognition processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # Place the webcam image on the background
        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Check if we are currently displaying student data
        if modeType == 1:
            elapsed_time = (datetime.now() - display_start_time).total_seconds()
            if elapsed_time < display_duration:
                # Continue displaying the current student's data
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]

                cv2.putText(imgBackground, str(studentInfo['total_presence']), (861, 125),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['class']), (1006, 550),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(id), (1006, 493),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                offset = (414 - w) // 2
                cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                print(f"Displaying student data for {id}, elapsed time: {elapsed_time} seconds")
                continue  # Skip new detections while displaying current student data

            # Reset after the display duration
            counter = 0
            modeType = 0
            studentInfo = []
            imgStudent = []
            current_displayed_id = None
            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
            print("Student data removed after 4 seconds")

        if faceCurFrame and modeType == 0:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    id = studentIds[matchIndex]

                    # Check if the student was recognized recently
                    if id in recognized_students:
                        last_recognized_time = recognized_students[id]
                        if datetime.now() - last_recognized_time < ignore_duration:
                            print(f"Ignoring recently recognized student {id}")
                            continue  # Ignore if recognized recently

                    # Update the time of recognition
                    recognized_students[id] = datetime.now()

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1
                        display_start_time = datetime.now()  # Start the display time
                        print(f"Loading student data for {id}")

                        studentInfo, imgStudent = get_student_data(id)
                        print(f"Student data loaded for {id}: {studentInfo}")

                        # Display the student data
                        datetimeObject = datetime.strptime(studentInfo['last_presence_time'], "%Y-%m-%d %H:%M:%S")
                        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                        if secondsElapsed > 30:
                            studentInfo['total_presence'] += 1
                            studentInfo['last_presence_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Update the JSON file with new data
                            with open('studentsData.json', 'w') as f:
                                json.dump({"Students": studentsData}, f, indent=4)
                            print(f"Updated presence for student {id}")

    # Display the background with the current data
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
