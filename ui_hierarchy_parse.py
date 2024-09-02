import importlib
import re
import sys
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Union
import config
from utils import Utility 

class UIHierarchyParser:
    def __init__(self, xml_file: str = 'ui_hierarchy.xml'):
        self.xml_file = xml_file
        self.criteria = {
            "class": [
                "android.widget.ImageView",
                "android.view.ViewGroup",
                "android.widget.Button",
                "android.widget.TextView",
                # "android.widget.EditText",
                "android.widget.CheckBox",
                # "android.widget.FrameLayout",
                "android.view.View",
                "android.widget.RadioButton",
                "android.widget.Spinner"
            ],
            "text": [
                "Get started",
                "Sign up",
                "Skip",
                "Enter Location Manually"
            ],
            "content-desc": [
                "Get started,Button",
                "Turn on Notification,Button"
            ]
        }

    def parse_ui_hierarchy(self) -> List[Dict[str, Union[str, Dict[str, int]]]]:
        """
        Parses the UI hierarchy XML file and extracts elements matching the specified criteria.
        
        :return: A list of elements with their bounding boxes and other attributes.
        """
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()

            elements_with_bounding_boxes = []
            for element in root.iter():
                if self._matches_criteria(element):
                    bounds = element.get('bounds')
                    if bounds:
                        coordinates = Utility.extract_coordinates(bounds)
                        coordinates.update({'node': ET.tostring(element, encoding='unicode')})
                        element_info = {
                            'class': element.get('class'),
                            'resource-id': element.get('resource-id'),
                            'text': element.get('text'),
                            'bounds': coordinates
                        }
                        elements_with_bounding_boxes.append(element_info)

            return elements_with_bounding_boxes

        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            return []
        
    def _matches_criteria(self, element: ET.Element) -> bool:
        """
        Checks if the given element matches the specified criteria.
        
        :param element: The XML element to check.
        :return: True if the element matches the criteria, False otherwise.
        """
        for key, values in self.criteria.items():
            if element.get(key) in values:
                return True
        return False
    

    @staticmethod
    def update_config_file(file_path, updates):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                for var, new_value in updates.items():
                    pattern = rf'^({var}\s*=\s*)(["\'])(.*?)(["\'])\s*$'
                    match = re.match(pattern, line)
                    if match:
                        new_line = f'{var} = {match.group(2)}{new_value}{match.group(4)}\n'
                        updated_lines.append(new_line)
                        break
                else:
                    updated_lines.append(line)

            for var, new_value in updates.items():
                pattern = rf'^({var}\s*=.*)$'
                if not any(re.match(pattern, line) for line in lines):
                    new_line = f'{var} = "{new_value}"\n'
                    updated_lines.append(new_line)

            with open(file_path, 'w') as file:
                file.writelines(updated_lines)

        except Exception as e:
            print(f"Error updating config file: {e}")
            sys.exit(1)

    @staticmethod
    def reload_config_module():
        try:
            import config
            importlib.reload(config)
        except Exception as e:
            print(f"Error reloading config module: {e}")
            sys.exit(1)

        
if __name__ == "__main__":
    parser = UIHierarchyParser()

    # Parse the XML to get elements with bounding boxes
    matching_elements = parser.parse_ui_hierarchy()

    # Example: Get activity name and related information
    activity, output_file, index, width, height = Utility.get_activity_name()

    updates = {
        'ACTIVITY': activity,
        'OUTPUT_FILE': output_file,
        'INDEX': index,
        'WIDTH': width,
        'HEIGHT': height  # Added HEIGHT as it was missing
    }

    # Update the config file and reload the config module
    parser.update_config_file( 'config.py',updates)
    parser.reload_config_module()
    # Print updated config values for verification
    print("ACTIVITY:", config.ACTIVITY)
    print("OUTPUT_FILE:", config.OUTPUT_FILE)

    # If matching elements are found, draw bounding boxes and save results
    if matching_elements:
        indexed_bounds =  Utility.draw_bounding_boxes('screen.png', matching_elements , output_file=output_file) 
        with open('indexed_bounds.json' ,'w') as f:
            json.dump(indexed_bounds, f)
    else:
        print("No matching elements found.")
        print("Please check the criteria and try again.")