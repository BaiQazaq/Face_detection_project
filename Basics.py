import cv2
import numpy as np 
import face_recognition

imgIsa = face_recognition.load_image_file('ImagesBasic/Isa.jpg')
imgIsa_test = face_recognition.load_image_file('ImagesBasic/Rock.jpg')

# Сохраните оригинальные цветные изображения
imgIsa_color = cv2.cvtColor(imgIsa, cv2.COLOR_BGR2RGB)
imgIsa_test_color = cv2.cvtColor(imgIsa_test, cv2.COLOR_BGR2RGB)

# Преобразуйте изображения в оттенки серого для face_location (по желанию)
imgIsa_gray = cv2.cvtColor(imgIsa, cv2.COLOR_BGR2GRAY)
imgIsa_test_gray = cv2.cvtColor(imgIsa_test, cv2.COLOR_BGR2GRAY)

faceLocation = face_recognition.face_locations(imgIsa_gray)[0]
encodeIsa = face_recognition.face_encodings(imgIsa_color)[0]

faceLocation_test = face_recognition.face_locations(imgIsa_test_gray)[0]
encodeIsa_test = face_recognition.face_encodings(imgIsa_test_color)[0]

# print("******", faceLocation) #this values are basicly top right and bottom left x1 y1 and x2 and y2
#(170, 526, 491, 205)
#print("******", faceLocation_test) #this values are basicly top right and bottom left x1 y1 and x2 and y2
# (357, 664, 911, 110)

#Раскомментируйте следующие строки, если хотите нарисовать прямоугольник на изображении в оттенках серого
cv2.rectangle(imgIsa_color, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)
cv2.rectangle(imgIsa_test_color, (faceLocation_test[3], faceLocation_test[0]), (faceLocation_test[1], faceLocation_test[2]), (255, 0, 255), 2)

#compare two images by encoding
results = face_recognition.compare_faces([encodeIsa], encodeIsa_test)
print(f"results = {results}")
# results = [True]   #result True show us that both encoding to be similar

# for finding how much they similar to each other we should find distance
# the lower distance the better match is 
faceDis = face_recognition.face_distance([encodeIsa], encodeIsa_test) #give us distance
# print(f"face distance = {faceDis}") #face distance = [0.45001624]

cv2.putText(imgIsa_test_color, f"{results} {round(faceDis[0], 2)}", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



cv2.imshow('Isa', imgIsa_color)
cv2.imshow('Isa_test', cv2.cvtColor(imgIsa_test_color, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)


