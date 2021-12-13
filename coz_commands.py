from PIL import ImageTk
from tkinter import *
import numpy as np
import pyttsx3
import pycozmo
import time
import PIL
import cv2


# Create the haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Last image, received from the robot.
last_im = []

# initialize speech synthesiser
synth = pyttsx3.init()
voices = synth.getProperty("voices")
synth.setProperty("voice", voices[3].id)
synth.setProperty("rate", 170)

cozmo_says = ''


def on_camera_image(_, new_im):
    """ Handle new images, coming from the robot. """
    global last_im
    last_im = new_im


def find_face(cli):
    # Raise head
    angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians - pycozmo.robot.MIN_HEAD_ANGLE.radians) / 2.0
    cli.set_head_angle(angle)

    # Register to receive new camera images
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    # Enable camera
    cli.enable_camera(color=True)

    # Create an instance of tkinter frame
    ws = Tk()
    ws.geometry("340x260")

    # Create a canvas
    canvas = Canvas(ws, width=320, height=240)
    canvas.pack()

    sliding = [(160, 120)] * 5
    last_slide = sliding[-1]
    while True:

        if last_im:

            # Get last image.
            img = np.array(last_im)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            face = [0] * 4
            for (x, y, w, h) in faces:
                face = [x, y, w, h] if w > face[2] else face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if sum(face):
                x, y, w, h = face
                midpoint = (x + h/2, y + w/2)
                sliding.pop(0)
                sliding.append(midpoint)

            # move to have face in centre of vision
            if last_slide != sliding[-1]:
                if sum([i[0] for i in sliding]) // 5 > 180:
                    cli.drive_wheels(lwheel_speed=40.0, rwheel_speed=-40.0, duration=0.1)
                elif sum([i[0] for i in sliding]) // 5 < 140:
                    cli.drive_wheels(lwheel_speed=-40.0, rwheel_speed=40.0, duration=0.1)

                if sum([i[1] for i in sliding]) // 5 > 140:
                    angle -= 0.02
                    cli.set_head_angle(angle)
                elif sum([i[1] for i in sliding]) // 5 < 100:
                    angle += 0.02
                    cli.set_head_angle(angle)

                last_slide = sliding[-1]

            pi = ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
            canvas.create_image(10, 10, anchor=NW, image=pi)
            canvas.update()


def say(cli, text):
    global cozmo_says
    cozmo_says = text

    synth.save_to_file(text, 'say_input.wav')
    synth.runAndWait()
    repeat(cli)


def repeat(cli):
    print(f'Cozmo: "{cozmo_says}"')
    cli.play_audio("say_input.wav")
    cli.wait_for(pycozmo.event.EvtAudioCompleted)


# everything past here is experimental


# Read the image
def locate_body(image):
    # image = cv2.imread(imagePath)
    # image = np.array(image_path)

    # Detect bodies in the image
    bodies = body_cascade.detectMultiScale(
        image,
        scaleFactor=1.02,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return bodies


def find_body(cli):
    # Create an instance of tkinter frame
    ws = Tk()
    ws.geometry("340x260")

    # Create a canvas
    canvas = Canvas(ws, width=320, height=240)
    canvas.pack()

    print('Cozmo is looking for a body...')
    # Raise head
    angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians - pycozmo.robot.MIN_HEAD_ANGLE.radians) / 2.0
    cli.set_head_angle(angle)

    # Register to receive new camera images.
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    # Enable camera.
    cli.enable_camera(color=True)

    # Run with 14 FPS. This is the frame rate of the robot camera.
    # timer = pycozmo.util.FPSTimer(2)
    while True:
        time.sleep(.5)

        if last_im:

            done = False
            bodies, image = None, None
            while not done:
                image = np.array(last_im)
                print(image.shape)
                bodies = locate_body(image)
                done = bool(len(bodies))

            for (x, y, w, h) in bodies:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Bodies found", image)
            cv2.waitKey(0)

            ws.destroy()
            cli.enable_camera(enable=False)
            print('Cozmo has looked for bodies...\n')
            break

            # pi = ImageTk.PhotoImage(last_im)
            # canvas.create_image(10, 10, anchor=NW, image=pi)
            # # ws.update()
            # canvas.update()
            # timer.sleep()
