import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


#calculate drawing_styles
def calculate_angle(a,b,c):
  a=np.array(a) #pinggul
  b=np.array(b) #tengah
  c=np.array(c)

  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle >180.0:
      angle = 360-angle

  return angle

counter = 0
stage = None
calo = 0

#timer
# def time_convert(sec):
#   mins = sec // 60
#   sec = sec % 60
#   hours = mins // 60
#   mins = mins % 60
#   print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

# For webcam input "0", for file input:name of the file
#cap = cv2.VideoCapture("Bike 1.mp4")
cap = cv2.VideoCapture("Bike2.mp4")
#cap = cv2.VideoCapture(0)

# picture's resolution check
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

print(height,width)

with mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as pose: #the higher the more confidence (depends on the body picture)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
      # continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #recolouring to RGB
    results = pose.process(image) # make detection

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #recolouring back

    #extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        #get coordinates pinggul, lutut, kaki
        pinggul = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        lutut = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        kaki = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        #calculate angle lutut
        angle = calculate_angle(pinggul, lutut, kaki) #sudut lutut
        #print (angle)
        #print(pinggul, lutut, kaki)
        #counter kaki
        if angle > 150:
            stage = "down"
        if angle < 130 and stage =='down':
            stage = "up"
            counter += 1
            calo += 0.097
            # print(counter)

    except:
        pass

    imageHeight, imageWidth, _ = image.shape

    for titik in mp_pose.PoseLandmark:
        titikBadan=results.pose_landmarks.landmark[titik]
        #pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(titikBadan.x, titikBadan.y, imageWidth, imageHeight)
        # print(titik)
        #print(pixelCoordinatesLandmark)
        # print(titikBadan)

    # calo = 0
    # if counter += 1:
    #     calo += 4.4

    #visualize sudut
    cv2.putText(image, str(angle), tuple(np.multiply(lutut, [1280,720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    #box counter
    cv2.rectangle(image,(0,0), (400,73), (245,117,16), -1)

    #counter no screen
    cv2.putText(image, 'Putaran', (15,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    #stage no screen
    cv2.putText(image, 'Stage', (105,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(stage), (100,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(image, 'Calorie', (255,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(round(calo,3)), (255,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)


    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(218,218,218), thickness=2, circle_radius=2)
        )

    # print  x y z
    #print(results.pose_landmarks)

    # display result
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
