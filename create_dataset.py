import cv2
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
dataset = 'dataset/'

if not os.path.exists(dataset):
    os.makedirs(dataset)

person_id = 1
count = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors = 5, minSize=(30,30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        count+=1
        cv2.imwrite(f'{dataset}Person-{person_id}-{count}.jpg', gray[y:y+h, x:x+w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 30:
        break

cap.release()
cv2.destroyAllWindows()