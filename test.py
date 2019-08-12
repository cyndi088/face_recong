import face_recognition
import cv2
import numpy as np

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
ipc_rtsp = 'rtsp://admin:a1234567@192.168.10.45:554'
video_capture = cv2.VideoCapture(ipc_rtsp)

# Load a sample picture and learn how to recognize it.
han_image = face_recognition.load_image_file("face_lib/han.jpg")
han_face_encoding = face_recognition.face_encodings(han_image)[0]

# Load a second sample picture and learn how to recognize it.
wu_image = face_recognition.load_image_file("face_lib/wu.jpg")
wu_face_encoding = face_recognition.face_encodings(wu_image)[0]

ding_image = face_recognition.load_image_file("face_lib/517141328540745172.jpg")
ding_face_encoding = face_recognition.face_encodings(ding_image)[0]

zheng_image = face_recognition.load_image_file("face_lib/20190809172226.jpg")
zheng_face_encoding = face_recognition.face_encodings(zheng_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [han_face_encoding,
                        wu_face_encoding,
                        ding_face_encoding,
                        zheng_face_encoding
                        ]
known_face_names = ["Han",
                    "Wu",
                    "Ding",
                    "Zheng"
                    ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(face_distances)
            print(face_distances[best_match_index])
            if face_distances[best_match_index] < 0.5:
                if matches[best_match_index]:
                    # best_match_index is numpt int64, need convert to python int
                    best_match_index = int(best_match_index)
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow(ipc_rtsp, frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()