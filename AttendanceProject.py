import cv2
import numpy as np 
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)
#print(f"mylist = {mylist}") #mylist = ['Abdulla.jpg', 'Isa.jpg', 'Malika.jpg', 'Diana.jpg']

for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}") # this code read the picture from the file and returne as multivariate array
                                        # and this allow show picture by method 'imshow()'
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) #this code grabe the name of persone from filename without expansion 'jpg'

print(f"Class name list = {classNames}")
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        encode = face_recognition.face_encodings(img_color)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")



encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)  # size of the image will divide to 4 or take just quarter of the image 
    imgSmall_color = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    imgSmall_gray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
    facesCurFrame = face_recognition.face_locations(imgSmall_gray)
    encodeCurFrame = face_recognition.face_encodings(imgSmall_color, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(f"faceDis = {faceDis}")
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4 #return size of image loc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img,(x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)




# faceLocation = face_recognition.face_locations(imgIsa_gray)[0]
# encodeIsa = face_recognition.face_encodings(imgIsa_color)[0]

# faceLocation_test = face_recognition.face_locations(imgIsa_test_gray)[0]
# encodeIsa_test = face_recognition.face_encodings(imgIsa_test_color)[0]


# imgIsa = face_recognition.load_image_file('ImagesBasic/Isa.jpg')
# imgIsa_test = face_recognition.load_image_file('ImagesBasic/Isa_test.jpg')

# # Сохраните оригинальные цветные изображения
# imgIsa_color = cv2.cvtColor(imgIsa, cv2.COLOR_BGR2RGB)
# imgIsa_test_color = cv2.cvtColor(imgIsa_test, cv2.COLOR_BGR2RGB)

# # Преобразуйте изображения в оттенки серого для face_location (по желанию)
# imgIsa_gray = cv2.cvtColor(imgIsa, cv2.COLOR_BGR2GRAY)
# imgIsa_test_gray = cv2.cvtColor(imgIsa_test, cv2.COLOR_BGR2GRAY)