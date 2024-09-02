# File: main.py

from utils import Utility
from PIL import Image
import os
import json
import google.generativeai as genai
import config
from caching import Cache
from ui_hierarchy_parse import UIHierarchyParser
if __name__ == "__main__":
    updates = {
        'DEBUG': False,
    }
    UIHierarchyParser.update_config_file('config.py', updates)
    UIHierarchyParser.reload_config_module()
    task = input("task: ")
    if not task.strip():
        print("task not entered")
        exit()

    activity_name = config.ACTIVITY

    # Check if the same task is found in cache storage; if yes, execute it
    if Cache.check_if_cached(task, activity_name, 'screen.png'):
        print("task found in cache and executed successfully")
    else:
        print("task not found in cache")
        imagePath = config.OUTPUT_FILE

        # Call LLM to identify the target element and action element
        genai.configure(api_key=config.GOOGLE_API_KEY)

        img = Image.open(imagePath)
        target_label, action, payload = Utility.model_response(task, img)

        is_proceed = input("Do you want to proceed with this response (y/n): ")
        if is_proceed == 'n':
            print("task not executed")
            # os.system("python ap.py")
            exit()

        change_label = input("Do you want to change the label number (y/n): ")
        if change_label == 'y':
            target_label = input("Enter the new label number: ")

        # Execute task
        json_data = {}
        with open("indexed_bounds.json") as f:
            json_data = json.load(f)

        coords = json_data[target_label]
        x_cord = str(coords["left"] + (coords["right"] - coords["left"]) / 2)
        y_cord = str(coords["top"] + (coords["bottom"] - coords["top"])  / 2)

        is_task_executed = Utility.execute_task(action, payload, x_cord=x_cord, y_cord=y_cord)

        # Cache task
        if is_task_executed:
            elements = []
            new_image_path = 'images/' + activity_name + '_' + str(config.INDEX) + '.png'
            Utility.draw_bounding_boxes('screen.png', elements, output_file=new_image_path)

            xml_path = 'xml_files/' + activity_name + '_' + str(config.INDEX) + '.xml'
            with open('ui_hierarchy.xml', 'r', encoding='utf-8') as file:
                data = file.read()

            with open(xml_path, 'w', encoding='utf-8') as file:
                file.write(data)

            Cache.cache_task(task, imagePath=new_image_path, element_coords=coords, activityName=activity_name, action=action, payload=payload, xml_path=xml_path)
        else:
            print("task couldn't be executed, so it is not cached")
