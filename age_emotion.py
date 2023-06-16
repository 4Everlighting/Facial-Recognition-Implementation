import face_recognition as fr, cv2, os, sys, math, face_recognition, numpy as np
from time import sleep
images_list = ["working_faces\\old_happy_man.jpg","working_faces\\young_sad_man.jpg"]

DEBUG_MODE = False
FONT_SIZE = 0.65

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def get_emotion_from_image(IMAGE):
    image = cv2.imread(IMAGE)
    from rmn import RMN
    m = RMN()
    results = m.detect_emotion_for_single_frame(image)
    emotion = results[0]['emo_label']
    if DEBUG_MODE:
      print(f'DEBUG: image "{IMAGE}" has emotion "{emotion}"')
    return emotion

def get_age_from_image(IMAGE):
  faceProto="opencv_face_detector.pbtxt"
  faceModel="opencv_face_detector_uint8.pb"
  ageProto="age_deploy.prototxt"
  ageModel="age_net.caffemodel"

  MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
  ageList=['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

  faceNet=cv2.dnn.readNet(faceModel,faceProto)
  ageNet=cv2.dnn.readNet(ageModel,ageProto)

  video=cv2.VideoCapture(IMAGE)
  padding=20
  while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
      cv2.waitKey()
      break

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
      print("No face detected")
      return None

    for faceBox in faceBoxes:
      face=frame[max(0,faceBox[1]-padding):
       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
       :min(faceBox[2]+padding, frame.shape[1]-1)]

    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    if DEBUG_MODE:
      print(f'Age: {age} years')
    return age

def draw_image_with_age_and_emotion(IMAGE,AGE,EMOTION):
  img = cv2.imread(IMAGE, 1)
  face_locations = face_recognition.face_locations(img)
  top = face_locations[0][0]
  right = face_locations[0][1]
  bottom = face_locations[0][2]
  left = face_locations[0][3]
  IMAGE_LABEL = f'Age: {AGE} Emo: {EMOTION}'
  cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
  cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
  font = cv2.FONT_HERSHEY_DUPLEX
  cv2.putText(img, IMAGE_LABEL, (left -20, bottom + 15), font, FONT_SIZE, (255, 255, 255), 2)
  while True:
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      return

def image_classifier(IMAGE):
  print(f"Determining age of image {IMAGE}")
  age = get_age_from_image(IMAGE)
  print(f"\tAge: {age}")
  print(f"Determining emotion of image {IMAGE}")
  emotion = get_emotion_from_image(IMAGE)
  print(f"\tEmotion: {emotion}")
  draw_image_with_age_and_emotion(IMAGE,age,emotion)


for i in images_list:
   image_classifier(i)
