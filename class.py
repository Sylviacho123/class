import cv2

# 載入 Haar 模型（確保這些 xml 檔與程式在同一資料夾）
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉灰階

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = len(faces)

    # 顯示出席人數
    cv2.putText(frame, f'Faces detected: {face_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # 繪製人臉方框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 偵測眼睛判斷是否清醒（有眼睛）或睡著（無眼睛）
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10)
        if len(eyes) >= 1:
            cv2.putText(frame, 'Awake', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Sleeping', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 偵測微笑判斷是否有笑
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'Not Smiling', (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    cv2.imshow('Smart Classroom Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
