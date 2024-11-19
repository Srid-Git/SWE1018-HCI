#Importing libraries
import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)       #Start video capture from live camera
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)      #Detect facial landmarks
screen_w, screen_h = pyautogui.size()       #width and height of computer screen

''' , , and .
Processes the RGB frame using the FaceMesh model to .
. '''

while True:
    _, frame = cam.read()       #Continuously captures frames from the camera
    frame = cv2.flip(frame, 1)      #flips them horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #converts the frame color from BGR to RGB
    output = face_mesh.process(rgb_frame)       #extract facial landmarks
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape       #height and width of captured frame

    if landmark_points:
        landmarks = landmark_points[0].landmark     #extracts the landmark points
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))       #calculates their screen positions and draws green circles on the detected points
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)        #move the mouse cursor on the screen to the corresponding position

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:         #check vertical distance between two landmarks is very small
            pyautogui.click()           # simulates a mouse click.
            pyautogui.sleep(1)
    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)