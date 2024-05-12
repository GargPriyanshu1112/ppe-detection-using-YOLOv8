# Import dependencies
import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser(
    description="Convert annotations from PascalVOC format to yolov8 format"
)
parser.add_argument(
    "-input_dir", type=str, metavar="", required=True, help="Path to images directory"
)
parser.add_argument(
    "-output_dir", type=str, metavar="", required=True, help="Path to output directory"
)
args = parser.parse_args()


def get_yolo_bbox_coords(x_min, y_min, x_max, y_max, img_h, img_w):
    x_center = ((x_max + x_min) / 2.0) / img_w
    y_center = ((y_max + y_min) / 2.0) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return (x_center, y_center, width, height)


def xml_to_txt(input_fpath, output_fpath, classes):
    tree = ET.parse(input_fpath)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)  # image width
    img_h = int(root.find("size/height").text)  # image height

    with output_fpath.open("w") as f:
        for object in root.iter("object"):
            # Assign object a class id
            object_name = object.find("name").text
            if object_name not in classes:
                continue
            class_id = classes.index(object_name)

            # Get pascalVOC bbox coordinates of the object
            x_min = int(object.find("bndbox/xmin").text)  # top-left x-coordinate
            y_min = int(object.find("bndbox/ymin").text)  # top-left y-coordinate
            x_max = int(object.find("bndbox/xmax").text)  # bottom-right x-coordinate
            y_max = int(object.find("bndbox/ymax").text)  # bottom-right y-coordinate

            # Convert pascalVOC bbox coordinates of the object to yolo format
            yolo_bbox = get_yolo_bbox_coords(x_min, y_min, x_max, y_max, img_h, img_w)

            # Save in YOLO format
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")


def convert(input_dir, output_dir, classes_file):
    if not output_dir.exists():
        os.mkdir(output_dir)

    classes = classes_file.read_text().splitlines()

    for xml_filepath in input_dir.iterdir():
        txt_filepath = output_dir / xml_filepath.with_suffix(".txt").name
        xml_to_txt(xml_filepath, txt_filepath, classes)


if __name__ == "__main__":
    CLASSES_FILE = Path(r"ppe_detection\data\classes.txt")
    convert(Path(args.input_dir), Path(args.output_dir), CLASSES_FILE)
