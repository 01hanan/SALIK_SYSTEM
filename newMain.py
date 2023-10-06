import math
import cv2
from supervision.draw.color import Color
from ultralytics import YOLO
import supervision as sv



def main():
    cap = cv2.VideoCapture(0)

    model = YOLO("best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness= 2,
        text_thickness= 2,
        text_scale= 1,
        text_color= Color.white(),
    )

    classNames = ["Pothole"]




    while True:
        ret, frame = cap.read()
        result = model(frame)[0]

        detections= sv.Detections.from_ultralytics(result)

        
        # labels = [
        #     f'{model.model.names[class_id]} {confidence:0.2f}'
        #     for _, confidence, class_id, _
        #     in detections
        # ]

        
        boxes = result.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            labels = [
                f"{classNames[cls]} {conf}"
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )



        cv2.imshow("YOLOv8", frame)





        if ( cv2.waitKey(30) == 27 ):
            break










if __name__ == "__main__":
    main()