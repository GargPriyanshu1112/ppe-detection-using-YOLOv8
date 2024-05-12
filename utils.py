# Import dependencies
import os
import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def get_class_distribution(xml_dirpath, save=False):
    class_counter = {}
    for xml_filepath in xml_dirpath.iterdir():
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        for object in root.iter("object"):
            class_ = object.find("name").text
            if class_ in ["glasses", "ear-protector", "safety-harness"]:
                print(class_)
            if class_ not in class_counter:
                class_counter[class_] = 0
            class_counter[class_] += 1

    labels, values = [], []
    for class_, count in class_counter.items():
        labels.append(class_)
        values.append(count)
    plt.bar(labels, values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    if save:
        plt.savefig("class_distribution.png")
    plt.show()


def get_random_sample(images_dirpath, labels_dirpath):
    # Get random image
    filename = random.choice(os.listdir(images_dirpath))
    image_filepath = os.path.join(images_dirpath, filename)
    image = cv2.imread(image_filepath)

    # Fetch coordinates of objects present in the image
    object_coord_list = []
    xml_filepath = os.path.join(labels_dirpath, filename.split(".jpg")[0] + ".xml")
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    for object in root.iter("object"):
        object_name = object.find("name").text
        x_min = int(object.find("bndbox/xmin").text)  # top-left x-coordinate
        y_min = int(object.find("bndbox/ymin").text)  # top-left y-coordinate
        x_max = int(object.find("bndbox/xmax").text)  # bottom-right x-coordinate
        y_max = int(object.find("bndbox/ymax").text)  # bottom-right y-coordinate
        object_coord_list.append([object_name, (x_min, y_min), (x_max, y_max)])

    # Display bboxes of objects present in the image
    for obj_name, top_left_pt, bottom_right_pt in object_coord_list:
        image = cv2.rectangle(image, top_left_pt, bottom_right_pt, (255, 0, 0), 2)
    cv2.imshow("Image with bboxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # images_dirpath = r"datasets\images"
    # xml_dirpath = r"datasets\labels"
    # get_random_sample(images_dirpath, xml_dirpath)

    # xml_dirpath = Path(r"datasets\labels")
    # get_class_distribution(xml_dirpath)
