import cv2

vid = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train.yml')
names = 0
id = 0

while True:
    flag, fream = vid.read()
    if flag:
        gray = cv2.cvtColor(fream,cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            id, _ = recognizer.predict(gray[y:y+h, x:x+w])

            cv2.rectangle(fream, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(fream, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(fream, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
        cv2.imshow("result", fream)
        cv2.waitKey(1)

vid.release()

cv2.destroyAllWindows()

