# Permissions
# from android.permissions import request_permissions, Permission, check_permission
# from time import sleep

# request_permissions([Permission.CAMERA])
# while check_permission(Permission.CAMERA) == False:
#     sleep(0.2)


from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.core.window import Window
import requests

from kivy.uix.camera import Camera
# import numpy as np
# from jnius import autoclass


# CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
labels = ["Amirul_Muminin", "Ihsanul_Ahsan", "Kiky_Rizkia", "Miftahul_Fahrina", "Muslimin"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --------- LBPH OPENCV -------------
model = cv2.face.LBPHFaceRecognizer_create()
model.read("lbph_model.yml")

# Window.size = (412,892)
# CAMERA_ID = CameraInfo.CAMERA_FACING_BACK
CAMERA_ID = CameraInfo.CAMERA_FACING_FRONT
CAMERA_RESOLUTION = (640, 480)


def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                (x0, y0 + baseline),  
                (max(xt, x0 + w), yt), 
                color, 
                2)
    cv2.rectangle(img,
                (x0, y0 - h),  
                (x0 + w, y0 + baseline), 
                color, 
                -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img


class MainWindow(Screen):
    pass


class SecondWindow(Screen):
    pass
    

class WindowManager(ScreenManager):
    pass


class AndroidCamera(Camera):
    resolution = CAMERA_RESOLUTION
    index = CAMERA_ID
    frame = None
    rected_frame = None


    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(AndroidCamera, self).on_tex(*l)
        self.texture = Texture.create(size=np.flip(self.resolution), colorfmt='rgb')
        self.frame = self.frame_from_buf()
        self.frame_to_screen(self.frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        rot = 1 if self.index else 3
        return np.rot90(frame_bgr, rot)

    def frame_to_screen(self, frame):
        if isinstance(self.rected_frame, np.ndarray):
            frame = self.rected_frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tostring()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')


class MainApp(MDApp):
    # __init__ is a special method in Python classes, it is the constructor method for a class.
    # The self is used to represent the instance of the class. With this keyword, you can 
    # access the attributes and methods of the class in python
    # *args(non keyboard argument) and *kwargs(keyboard argument) are special keyword which allows 
    # function to take variable length argument.

    def __init__(self, **kwargs):
        self.title = "Viola Jones object detection framework"
        
        # The super() function is used to give access to methods and properties of a parent or sibling class.
        super().__init__(**kwargs)

    def build(self):
        return Builder.load_file("main.kv")

    def back_to_home_screen(self, x="", y=""):
        self.root.current = "first"

        # Stop camera and processing
        Clock.unschedule(self.update)

        if self.dialog:
            self.dialog.dismiss(force=True)
        else:
            print("NO DIALOGUE BOX")

    def show_alert_dialog(self, label_text):
        close_button = MDFlatButton(
                text='Close',
                on_touch_up=self.back_to_home_screen)
        self.dialog = MDDialog(
            title="Logged In Successful",text=label_text, 
            buttons=[close_button])
        self.dialog.open()

    def start_camera(self):
        self.root.screens[1].ids.vid.play = True
        Clock.schedule_interval(self.update, 1/30)

    def update(self, dt):
        camera = self.root.screens[1].ids.vid

        if isinstance(camera.frame, np.ndarray):
            gray = cv2.cvtColor(camera.frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces):
                frame_copy = camera.frame.copy()
                face = faces[np.argmax(faces[:, 3])]
                x, y, w, h = face
                roi = gray[y:y+h, x:x+w]
                resized_roi = cv2.resize(roi, (100, 100))
                idx, confidence = model.predict(resized_roi)
                confidence = confidence

                if confidence >= 99 and confidence <= 100:
                    label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                    r = requests.put('http://test.teknikaja.my.id/1', json={'state': 'OPEN'})
                    toast('Logged In Successful')
                    camera.play = False
                    camera.frame = None
                    Clock.unschedule(self.update)
                    self.root.current = "first"
                    self.show_alert_dialog(label_text)
                else:
                    label_text = "Unknown"

                camera.rected_frame = draw_ped(frame_copy, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))
            else:
                camera.rected_frame = None


if __name__ == '__main__':
    MainApp().run()