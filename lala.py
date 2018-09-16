import cv2,os
import sqlite3
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from PIL import Image
import pickle

class Widgets(Widget):
    def __init__(self, **kwargs):
        super(Widgets, self).__init__(**kwargs)

    def Login(self):
        ceye="haarcascade_eye.xml"
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        path = 'dataSet/face'
        
        conn=sqlite3.connect('database.db')
        c=conn.cursor()
        def read_db(idd):
            c.execute('SELECT * FROM DATABASE WHERE ID=1')
            data=c.fetchall()
            print(data)
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
        while True:
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
            faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
                cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
                read_db(nbr_predicted)
                #cv2.putText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255)
                cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x, 50),font, 1.0, (255, 255, 0), lineType=cv2.LINE_AA)
                #Draw the text
                cv2.imshow('im',im)
                cv2.waitKey(10)





    def Train(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create();
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        path = 'faceR/face'
        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            # images will contains face images
            images = []
            # labels will contains the label that is assigned to the image
            labels = []
            for image_path in image_paths:
                # Read the image and convert to grayscale
                image_pil = Image.open(image_path).convert('L')
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')
                # Get the label of the image
                nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face", ""))
                #name = str(os.path.split(image_path).split()[2]
                #nbr=int(''.join(str(ord(c)) for c in nbr))
                print(nbr)
                #  print(name)
                # Detect the face in the image
                faces = faceCascade.detectMultiScale(image)
                # If face is detected, append the face to images and the label to labels
                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(nbr)
                    cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                    cv2.waitKey(10)
                # return the images list and labels list
            return images, labels
        images, labels = get_images_and_labels('faceR/face')
        cv2.imshow('test',images[0])
        cv2.waitKey(1)

        recognizer.train(images, np.array(labels))
        recognizer.save('trainer1.yml')

    def Register(self):
        cam = cv2.VideoCapture(0)
        detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        i=0
        con=sqlite3.connect('data.db')
        offset=50
        name=input('enter your id')
        ids=input('enter your name')
        res = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
        f=1
        for tab in res:
            if tab[0]=='nae':
                f=0
        if f==1:
            con.execute('CREATE TABLE nae(ID NUMBER PRIMARY KEY,NAME VARCHAR[20])')
        con.execute('INSERT INTO nae VALUES(?,?)',[name,ids])
        con.commit()
        con.close()

        while True:
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
            faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                i=i+1
                cv2.imwrite("dataSet/face/"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
                cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
                cv2.imshow('im',im[y:y+h+offset,x:x+w+offset])
                cv2.waitKey(1)
            if i>15:
                cam.release()
                cv2.destroyAllWindows()
                break



class MainApp(App):
    def build(self):
        return Widgets()

if __name__ =="__main__":
    MainApp().run()
