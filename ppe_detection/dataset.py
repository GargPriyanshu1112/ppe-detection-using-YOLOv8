# Import dependencies
import os
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET


# Paths where new dataset will be saved
PPE_IMG_DIRPATH = r"test\data\images"
PPE_ANNOT_DIRPATH = r"test\data\xml_files"


def save_annotations_in_xml(filename, object_coord_pairs, image_size):
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = filename + ".jpg"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])

    for name, (x_min, y_min, x_max, y_max) in object_coord_pairs:
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = name
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x_min)
        ET.SubElement(bndbox, "ymin").text = str(y_min)
        ET.SubElement(bndbox, "xmax").text = str(x_max)
        ET.SubElement(bndbox, "ymax").text = str(y_max)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(PPE_ANNOT_DIRPATH, filename + ".xml"))  # TODO: add location


def extract_person_data(image_path, object_coord_pairs):
    image = Image.open(image_path)

    for idx, (name, (x_min, y_min, x_max, y_max)) in enumerate(object_coord_pairs):
        if name == "person":
            # Crop the bbox
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_size = cropped_image.size  # (width, height)

            # Search for objects inside the cropped image
            cropped_image_object_coord_pairs = []
            for name, (obj_xmin, obj_ymin, obj_xmax, obj_ymax) in object_coord_pairs:
                # If the object falls inside the cropped image, refactor the coordinates.
                # Change them relative to the cropped image.
                if (
                    (name != "person")
                    and (x_min <= obj_xmin <= x_max)
                    and (x_min <= obj_xmax <= x_max)
                    and (y_min <= obj_ymin <= y_max)
                    and (y_min <= obj_ymax <= y_max)
                ):
                    obj_xmin -= x_min
                    obj_ymin -= y_min
                    obj_xmax -= x_min
                    obj_ymax -= y_min

                    cropped_image_object_coord_pairs.append(
                        [name, (obj_xmin, obj_ymin, obj_xmax, obj_ymax)]
                    )
            # Save cropped image and the annotations of objects that fall inside it
            filename = image_path.with_suffix("").name
            cropped_image.save(
                f"{PPE_IMG_DIRPATH}\cropped_{filename}_{idx}.jpg"
            )  # TODO: add location
            save_annotations_in_xml(
                f"cropped_{filename}_{idx}",
                cropped_image_object_coord_pairs,
                cropped_size,
            )


def get_all_object_coord_pairs(xml_filepath):
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    # Loop through each object in the image
    object_coord_pairs = []
    for object in root.iter("object"):
        name = object.find("name").text
        x_min = int(object.find("bndbox/xmin").text)  # top-left x-coordinate
        y_min = int(object.find("bndbox/ymin").text)  # top-left y-coordinate
        x_max = int(object.find("bndbox/xmax").text)  # bottom-right x-coordinate
        y_max = int(object.find("bndbox/ymax").text)  # bottom-right y-coordinate
        object_coord_pairs.append([name, (x_min, y_min, x_max, y_max)])
    return object_coord_pairs


def prepare_dataset(IMAGES_DIRPATH, XML_DIRPATH):
    if not os.path.exists(PPE_IMG_DIRPATH):
        os.makedirs(PPE_IMG_DIRPATH)
    if not os.path.exists(PPE_ANNOT_DIRPATH):
        os.makedirs(PPE_ANNOT_DIRPATH)

    for xml_filepath in XML_DIRPATH.iterdir():
        image_path = IMAGES_DIRPATH / xml_filepath.with_suffix(".jpg").name
        object_coord_pairs = get_all_object_coord_pairs(xml_filepath)
        extract_person_data(image_path, object_coord_pairs)


if __name__ == "__main__":
    IMAGES_DIRPATH = Path("datasets\images")
    XML_DIRPATH = Path("datasets\labels")

    prepare_dataset(IMAGES_DIRPATH, XML_DIRPATH)
