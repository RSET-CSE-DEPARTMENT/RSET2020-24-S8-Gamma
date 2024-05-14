import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectDetector:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO('best.pt')
        self.annotator = None

    def predict(self, im0):
        results = self.model(im0)
        return results

    def plot_bboxes(self, results, im0):
        self.annotator = Annotator(im0, 3, results.names)
        boxes = results.boxes.xyxy.cpu()
        clss = results.boxes.cls.cpu().tolist()
        for box, cls in zip(boxes, clss):
            self.annotator.box_label(box, label=results.names[int(cls)], color=colors(int(cls), True))
        return im0

    def process_frame(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while True:
            ret, im0 = cap.read()
            if not ret:
                break
            
            results = self.predict(im0)
            annotated_image = self.plot_bboxes(results, im0)

            yield cv2.imencode('.jpg', annotated_image)[1].tobytes()
