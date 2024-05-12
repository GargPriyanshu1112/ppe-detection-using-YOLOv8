# Import dependencies
import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument(
    "-input_dir", type=str, metavar="", required=True, help="Path to images directory"
)
parser.add_argument(
    "-output_dir", type=str, metavar="", required=True, help="Path to output directory"
)
parser.add_argument(
    "-person_det_model",
    type=str,
    metavar="",
    required=True,
    help="Path to person detection model weights",
)
parser.add_argument(
    "-ppe_detection_model",
    type=str,
    metavar="",
    required=True,
    help="Path to ppe detection model weights",
)
args = parser.parse_args()


class Test:
    def __init__(
        self,
        input_dir,
        output_dir,
        person_det_model_weights,
        ppe_detection_model_weights,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.person_det_model = YOLO(person_det_model_weights)
        self.ppe_detection_model = YOLO(ppe_detection_model_weights)
        self.res_dir = None

    def bbox_is_corrupted(self, xyxy):
        bbox_w = xyxy[2] - xyxy[0]
        bbox_h = xyxy[3] - xyxy[1]
        return (bbox_h < 10) or (bbox_w < 10)

    def detect_person(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        # Get predicted bboxes info
        boxes = self.person_det_model.predict(source=[image])[0].boxes

        image_with_bboxes = image.copy()
        num_persons_detected = len(boxes.cls)

        if num_persons_detected > 0:
            self.res_dir = self.output_dir / image_path.with_suffix("").name
            os.mkdir(self.res_dir)

            for i in range(num_persons_detected):
                xyxy = boxes.xyxy[i]  # top-left and bottom-right coords of bbox
                conf = round(float(boxes.conf[i]), 1)  # prediction confidence
                # Draw bbox
                cv2.rectangle(
                    image_with_bboxes,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (255, 0, 0),
                    2,
                )
                # Display prediction confidence
                cv2.putText(
                    image_with_bboxes,
                    str(conf),
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                # Crop the bbox
                cropped_bbox = image[
                    int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                ]

                self.ppe_detection(cropped_bbox, i)
        # Save image
        cv2.imwrite(self.res_dir / f"person_det.jpg", image_with_bboxes)

    def ppe_detection(self, image, i):
        # Get predicted bboxes info
        boxes = self.ppe_detection_model.predict(source=[image])[0].boxes

        image_with_bboxes = image.copy()
        num_ppe_items = len(boxes.cls)

        if num_ppe_items > 0:
            for idx in range(num_ppe_items):
                xyxy = boxes.xyxy[idx]  # top-left and bottom-right coords of bbox
                conf = round(float(boxes.conf[idx]), 1)  # prediction confidence

                if self.bbox_is_corrupted(xyxy):
                    continue

                # Draw bbox
                cv2.rectangle(
                    image_with_bboxes,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (255, 0, 0),
                    2,
                )
                # Display prediction confidence
                cv2.putText(
                    image_with_bboxes,
                    str(conf),
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
        # Save image
        cv2.imwrite(self.res_dir / f"P_#{i+1}_ppe_det.jpg", image_with_bboxes)

    def predict(self):
        if not self.output_dir.exists():
            os.mkdir(self.output_dir)

        for image_path in self.input_dir.iterdir():
            self.detect_person(image_path)


if __name__ == "__main__":
    test = Test(
        args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model
    )
    test.predict()

# python inference.py -input_dir="test/test_images" -output_dir="test/output_files" -person_det_model="test/weights/person_det_model_weights.pt" -ppe_detection_model="test/weights/ppe_detection_model_weights.pt"
