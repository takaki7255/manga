import xml.etree.ElementTree as ET
import re
import os


def get_xml_files_from_folder(folder):
    """Get the XML files from a folder."""
    xml_files = []
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            xml_files.append(os.path.join(folder, file))
        # sort by alphabet
        xml_files.sort()
    return xml_files


def compute_shortest_frame_length(frame):
    """Compute the shortest length of a frame based on its coordinates."""
    width = abs(int(frame["xmax"]) - int(frame["xmin"]))
    height = abs(int(frame["ymax"]) - int(frame["ymin"]))
    return min(width, height)


def extract_frame_coordinates(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Initialize empty dictionary to store results
    page_frame_mapping = {}

    # Initialize a variable to keep track of the current page index
    current_page_index = None

    # Loop through each element in the XML
    for elem in root.iter():
        # Check if the element is a <page> tag
        if elem.tag == "page":
            current_page_index = elem.attrib["index"]
            page_frame_mapping[current_page_index] = []

        # Check if the element is a <frame> tag
        elif elem.tag == "frame" and current_page_index is not None:
            coords = {
                "id": elem.attrib["id"],
                "xmin": elem.attrib["xmin"],
                "ymin": elem.attrib["ymin"],
                "xmax": elem.attrib["xmax"],
                "ymax": elem.attrib["ymax"],
            }
            page_frame_mapping[current_page_index].append(coords)

    return page_frame_mapping


def extract_shortest_frame_length_per_page(xml_file_path):
    # First, get the frame coordinates for each page
    page_frame_mapping = extract_frame_coordinates(xml_file_path)

    # Compute the shortest frame length for each page
    page_shortest_length = {}
    for page, frames in page_frame_mapping.items():
        shortest_lengths = [compute_shortest_frame_length(frame) for frame in frames]
        if shortest_lengths:  # Ensure the list is not empty
            page_shortest_length[page] = min(shortest_lengths)
        else:
            page_shortest_length[page] = None  # No frames for this page

    return page_shortest_length


def find_shortest_frame_in_manga(xml_file_path):
    # First, get the shortest frame length for each page
    shortest_frame_lengths = extract_shortest_frame_length_per_page(xml_file_path)

    # Filter out pages that have no frames (value is None)
    valid_shortest_lengths = {
        k: v for k, v in shortest_frame_lengths.items() if v is not None
    }

    # Find the page with the absolute shortest frame length
    shortest_page = min(valid_shortest_lengths, key=valid_shortest_lengths.get)
    shortest_length = valid_shortest_lengths[shortest_page]

    # Get the frame with the shortest length from the selected page
    page_frames = extract_frame_coordinates(xml_file_path)[shortest_page]
    shortest_frame = min(page_frames, key=compute_shortest_frame_length)

    return {"page": shortest_page, "frame": shortest_frame, "length": shortest_length}


# Example usage
if __name__ == "__main__":
    xml_files = get_xml_files_from_folder(
        "./../Manga109_released_2021_12_30/annotations.v2020.12.18"
    )
    for xml_file in xml_files:
        print(xml_file)
        result = find_shortest_frame_in_manga(xml_file)
        print(result)
        with open("shortest_frame.txt", "a") as f:
            f.write(xml_file + "\n")
            f.write(str(result) + "\n")
            f.write("\n")
    # # belmondo以降のみファイル名として取得
    # manga = xml_file_path.split("/")[-1].split(".")[0]
    # result = find_shortest_frame_in_manga(xml_file_path)
    # print(manga)
    # print(result)
