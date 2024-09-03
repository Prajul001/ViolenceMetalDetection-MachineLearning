import numpy as np
import cv2
import os
import datetime
import pygame  # Import pygame for sound
from collections import deque
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as KerasDepthwiseConv2D
import serial
import telepot

# Option to handle model loading issues
class CustomDepthwiseConv2D(KerasDepthwiseConv2D):
    def __init__(self, **kwargs):
        super(CustomDepthwiseConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super(CustomDepthwiseConv2D, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return cls(**config)

def play_alarm_sound():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(r"") # Alarm Music Path
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

def print_results(video_path, model_path, output_dir='output', threshold=0.50, buffer_size=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        print("Loading model...")
        model = load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    Q = deque(maxlen=buffer_size)
    vs = cv2.VideoCapture(video_path)
    writer = None
    frame_count = 0
    W, H = None, None

    if not vs.isOpened():
        print("Error: Video file could not be opened.")
        return

    alarm_played = False  # Flag to track if the alarm has been played

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            print("Frame not grabbed. Exiting...")
            break

        if W is None or H is None:
            H, W = frame.shape[:2]

        output = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32") / 255
        frame_reshaped = frame_resized.reshape(128, 128, 3)

        try:
            preds = model.predict(np.expand_dims(frame_reshaped, axis=0))[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        label = (results > threshold)[0]

        text_color = (0, 255, 0)  # Default color: green
        text = "Violence: False"
        if label:
            text_color = (0, 0, 255)  # Red for violence
            text = "Violence: True"
            if not alarm_played:
                print("Violence Detected")
                play_alarm_sound()  # Play alarm sound
                alarm_played = True

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        if label and writer is None:
            date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_output_path = os.path.join(output_dir, f'{date_string}.avi')
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_output_path, fourcc, 30, (W, H), True)
            print(f"Video writer initialized and saving to: {video_output_path}")

            # Capture a frame for saving
            frame_name = f'Frame{frame_count}.jpg'
            print(f'Capturing --- {frame_name}')
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            frame_count += 1

        if writer:
            writer.write(output)

        cv2.imshow("Video Display", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    if writer:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()

def metal_detection(serial_port, token, receiver_id):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    serialcom = serial.Serial(serial_port, 9600)
    serialcom.timeout = 1

    cap = cv2.VideoCapture(0)
    i = 0
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%I.%M.%S%p_%A")
    time_str = date_string[10:17]
    out = cv2.VideoWriter(time_str + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (640, 480))

    while True:
        a = serialcom.readline(35).decode('utf-8').rstrip()
        if a == "1":
            serialcom.close()
            z = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.resize(frame, (640, 480))
                boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

                for (xA, yA, xB, yB) in boxes:
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
                dt = str(datetime.datetime.now())
                cv2.putText(frame, dt, (10, 100), font, 1, (0, 255, 0), 2, cv2.LINE_8)

                cv2.imwrite('Frame' + str(i) + '.jpg', frame)
                out.write(frame)

                cv2.imshow('Original', frame)
                if z == 1:
                    bot = telepot.Bot(token)
                    bot.sendMessage(receiver_id, 'Metal Detected')
                    bot.sendPhoto(receiver_id, photo=open('Frame0.jpg', 'rb'))
                    z = 2

                if cv2.waitKey(1) & 0xFF == ord('a'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Violence Detection/2. Metal Detection")
    choice = input("Enter Your Choice:")
    if choice == '1':
        video_path = r"" # Video Path
        model_path = r"" # Model Path
        print_results(video_path, model_path)
    elif choice == '2':
        serial_port = 'COM3'
        token = ''  # Provide your telegram token here
        receiver_id = ''  # Provide the receiver ID here
        metal_detection(serial_port, token, receiver_id)
    else:
        print("Invalid choice. Please enter '1' for Violence Detection or '2' for Metal Detection.")
