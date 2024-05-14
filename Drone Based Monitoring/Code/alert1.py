import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

password = "password"
from_email = "from@gmail.com" 
to_email = "to@gmail.com"
server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

def send_email(to_email, from_email, object_detected=1):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"

    message_body = f'ALERT - red strawberry has been detected!!'

    message.attach(MIMEText(message_body, 'plain'))
    server.sendmail(from_email, to_email, message.as_string())

class ObjectDetection:
    # strawberry = {'Early turning','Green','Late Turning','Red','Turning','White'}
    def __init__(self, capture_source):

        self.capture_source = capture_source
        self.email_sent = False


        self.model = YOLO("best.pt")


        self.annotator = None
        self.start_time = 0
        self.end_time = 0


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_source)
        # stream_url = 'tcp/h264://172.20.10.3:8554/'
        # cap = cv2.VideoCapture(stream_url)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_count = 0

        # while True:
        #     ret, im0 = cap.read()

        #     if not ret:
        #         print("Error: Unable to read frame.")
        #         break

        #     results = self.predict(im0)
        #     im0, class_ids = self.plot_bboxes(results, im0)

        #     if 3.0 in class_ids:
        #         if not self.email_sent:
        #             send_email(to_email, from_email, len(class_ids))
        #             self.email_sent = True
        #     else:
        #         self.email_sent = False

        #     self.display_fps(im0)
        #     cv2.imshow('YOLOv8 Detection', im0)

        #     if cv2.waitKey(1) & 0xFF == 27:
        #         break

        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            print('class id',class_ids)

            if 3.0 in class_ids:
                if not self.email_sent:
                    send_email(to_email, from_email, len(class_ids))
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_source=0)
detector()
