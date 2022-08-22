import face_recognition
import cv2
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)

Gaurav_image = face_recognition.load_image_file("Gaurav.png")
Gaurav_face_encoding = face_recognition.face_encodings(Gaurav_image)[0]

Pushpraj_image = face_recognition.load_image_file("Pushpraj.png")
Pushpraj_face_encoding = face_recognition.face_encodings(Pushpraj_image)[0]

Mahima_image = face_recognition.load_image_file("Mahima.png")
Mahima_face_encoding = face_recognition.face_encodings(Mahima_image)[0]

Sardeep_image = face_recognition.load_image_file("Sardeep.jpg")
Sardeep_face_encoding = face_recognition.face_encodings(Sardeep_image)[0]

Sandeep_image = face_recognition.load_image_file("Sandeep.png")
Sandeep_face_encoding = face_recognition.face_encodings(Sandeep_image)[0]

Aakash_image = face_recognition.load_image_file("Aakash.png")
Aakash_face_encoding = face_recognition.face_encodings(Aakash_image)[0]

Abhishek_image = face_recognition.load_image_file("Abhishek.png")
Abhishek_face_encoding = face_recognition.face_encodings(Abhishek_image)[0]

Aishwarya_image = face_recognition.load_image_file("Aishwarya.png")
Aishwarya_face_encoding = face_recognition.face_encodings(Aishwarya_image)[0]

Akhil_image = face_recognition.load_image_file("Akhil.png")
Akhil_face_encoding = face_recognition.face_encodings(Akhil_image)[0]

Akhilesh_image = face_recognition.load_image_file("Akhilesh.png")
Akhilesh_face_encoding = face_recognition.face_encodings(Akhilesh_image)[0]

Anil_image = face_recognition.load_image_file("Anil.png")
Anil_face_encoding = face_recognition.face_encodings(Anil_image)[0]

Ankit_image = face_recognition.load_image_file("Ankit.png")
Ankit_face_encoding = face_recognition.face_encodings(Ankit_image)[0]

Ankita_image = face_recognition.load_image_file("Ankita.png")
Ankita_face_encoding = face_recognition.face_encodings(Ankita_image)[0]

AnkitE_image = face_recognition.load_image_file("AnkitE.png")
AnkitE_face_encoding = face_recognition.face_encodings(AnkitE_image)[0]

AnkitP_image = face_recognition.load_image_file("AnkitP.png")
AnkitP_face_encoding = face_recognition.face_encodings(AnkitP_image)[0]

aprajita_image = face_recognition.load_image_file("aprajita.png")
aprajita_face_encoding = face_recognition.face_encodings(aprajita_image)[0]

Ayush_image = face_recognition.load_image_file("Ayush.png")
Ayush_face_encoding = face_recognition.face_encodings(Ayush_image)[0]

Bhavesh_image = face_recognition.load_image_file("Bhavesh.png")
Bhavesh_face_encoding = face_recognition.face_encodings(Bhavesh_image)[0]

Chitresh_image = face_recognition.load_image_file("Chitresh.png")
Chitresh_face_encoding = face_recognition.face_encodings(Chitresh_image)[0]

Deepak_image = face_recognition.load_image_file("Deepak.png")
Deepak_face_encoding = face_recognition.face_encodings(Deepak_image)[0]

Divyanshi_image = face_recognition.load_image_file("Divyanshi.png")
Divyanshi_face_encoding = face_recognition.face_encodings(Divyanshi_image)[0]

Harsh_image = face_recognition.load_image_file("Harsh.png")
Harsh_face_encoding = face_recognition.face_encodings(Harsh_image)[0]

Jighyasa_image = face_recognition.load_image_file("Jighyasa.png")
Jighyasa_face_encoding = face_recognition.face_encodings(Jighyasa_image)[0]

Krishna_image = face_recognition.load_image_file("Krishna.png")
Krishna_face_encoding = face_recognition.face_encodings(Krishna_image)[0]

Mitesh_image = face_recognition.load_image_file("Mitesh.png")
Mitesh_face_encoding = face_recognition.face_encodings(Mitesh_image)[0]

Neha_image = face_recognition.load_image_file("Neha.png")
Neha_face_encoding = face_recognition.face_encodings(Neha_image)[0]

Nisha_image = face_recognition.load_image_file("Nisha.png")
Nisha_face_encoding = face_recognition.face_encodings(Nisha_image)[0]

Nishi_image = face_recognition.load_image_file("Nishi.png")
Nishi_face_encoding = face_recognition.face_encodings(Nishi_image)[0]

Nitin_image = face_recognition.load_image_file("Nitin.png")
Nitin_face_encoding = face_recognition.face_encodings(Nitin_image)[0]

Rahul_image = face_recognition.load_image_file("Rahul.png")
Rahul_face_encoding = face_recognition.face_encodings(Rahul_image)[0]

Sahil_image = face_recognition.load_image_file("Sahil.png")
Sahil_face_encoding = face_recognition.face_encodings(Sahil_image)[0]

Shikha_image = face_recognition.load_image_file("Shikha.png")
Shikha_face_encoding = face_recognition.face_encodings(Shikha_image)[0]

Shivam_image = face_recognition.load_image_file("Shivam.png")
Shivam_face_encoding = face_recognition.face_encodings(Shivam_image)[0]

Shivasha_image = face_recognition.load_image_file("Shivasha.png")
Shivasha_face_encoding = face_recognition.face_encodings(Shivasha_image)[0]

Shriya_image = face_recognition.load_image_file("Shriya.png")
Shriya_face_encoding = face_recognition.face_encodings(Shriya_image)[0]

ShriyaA_image = face_recognition.load_image_file("ShriyaA.png")
ShriyaA_face_encoding = face_recognition.face_encodings(ShriyaA_image)[0]

Shruti_image = face_recognition.load_image_file("Shruti.png")
Shruti_face_encoding = face_recognition.face_encodings(Shruti_image)[0]

ShubhamJ_image = face_recognition.load_image_file("ShubhamJ.png")
ShubhamJ_face_encoding = face_recognition.face_encodings(ShubhamJ_image)[0]

Shubhankit_image = face_recognition.load_image_file("Shubhankit.png")
Shubhankit_face_encoding = face_recognition.face_encodings(Shubhankit_image)[0]

Sonika_image = face_recognition.load_image_file("Sonika.png")
Sonika_face_encoding = face_recognition.face_encodings(Sonika_image)[0]

Swati_image = face_recognition.load_image_file("Swati.png")
Swati_face_encoding = face_recognition.face_encodings(Swati_image)[0]

Vikalp_image = face_recognition.load_image_file("Vikalp.png")
Vikalp_face_encoding = face_recognition.face_encodings(Vikalp_image)[0]

Vinitha_image = face_recognition.load_image_file("Vinitha.png")
Vinitha_face_encoding = face_recognition.face_encodings(Vinitha_image)[0]

Soumya_image = face_recognition.load_image_file("Soumya.png")
Soumya_face_encoding = face_recognition.face_encodings(Soumya_image)[0]




known_face_encodings = [
    Gaurav_face_encoding,
    Mahima_face_encoding,
    Pushpraj_face_encoding,
    Sardeep_face_encoding,
    Sandeep_face_encoding,
    Aakash_face_encoding,
    Abhishek_face_encoding,
    Aishwarya_face_encoding,
    Akhil_face_encoding,
    Akhilesh_face_encoding,
    Anil_face_encoding,
    Ankit_face_encoding,
    Ankita_face_encoding,
    AnkitE_face_encoding,
    AnkitP_face_encoding,
    aprajita_face_encoding,
    Ayush_face_encoding,
    Bhavesh_face_encoding,
    Chitresh_face_encoding,
    Deepak_face_encoding,
    Divyanshi_face_encoding,
    Harsh_face_encoding,
    Jighyasa_face_encoding,
    Krishna_face_encoding,
    Mitesh_face_encoding,
    Neha_face_encoding,
    Nisha_face_encoding,
    Nishi_face_encoding,
    Nitin_face_encoding,
    Rahul_face_encoding,
    Sahil_face_encoding,
    Shivam_face_encoding,
    Shivasha_face_encoding,
    Shriya_face_encoding,
     ShriyaA_face_encoding,
     Shruti_face_encoding,
     ShubhamJ_face_encoding,
     Shubhankit_face_encoding,
     Sonika_face_encoding,
     Swati_face_encoding,
     Vikalp_face_encoding,
     Vinitha_face_encoding,
     Soumya_face_encoding

]
known_face_names = [
    "Gaurav Gaur",
    "Mahima",
    "Pushpraj",
    "Sardeep",
    "Sandeep",
    "Aakash",
    "Abhishek",
    "Aishwarya",
    "Akhil Verma",
    "Akhilesh",
    "Anil",
    "Ankit",
    "Ankita",
    "Ankit Engle",
    "Ankit Pawar",
    "Aparajita",
    "Ayush",
    "Bhavesh",
    "Chitresh",
    "Deepak",
    "Divyanshi",
    "Harsh",
    "Jighaysa",
    "Krishna",
    "Mitesh",
    "Neha",
    "Nisha",
    "Nishi",
    "Nitin",
    "Rahul",
    "Sahil",
    "Shivam",
    "Shivasha",
    "Shriya",
    "Shriya Agnihotri",
    "Shruti",
    "Shubham Joshi",
    "Shubhankit",
    "Sonika",
    "Swati",
    "Vikalp",
    "Vinitha",
    "Soumya"
]
List_Name={}
List_time=set()
face_locations = []
face_encodings = []
face_names = []
login={}
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    

    if process_this_frame:
       
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        count=0
        for face_encoding in face_encodings:
         
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
              

            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
               
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                name = known_face_names[best_match_index]
                List_Name[name]= now.strftime("%H:%M:%S")
                login.update(List_Name)
                
                
                
                
            face_names.append(name)
            
            

            
                

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

                
length=len(List_Name)   
print("Login is ",List_Name,"total is",length)



