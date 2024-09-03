import json
import os
import re
import sys
import importlib
from PIL import Image, ImageDraw, ImageFont
from ppadb.client import Client as AdbClient
import uiautomator2 as u2
import google.generativeai as genai

class Utility:

    @staticmethod
    def draw_bounding_boxes2(image_path, bounds , output_file_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([bounds['left'], bounds['top'], bounds['right'], bounds['bottom']], outline='green', width=2)
        
        #return image with bounding box save in 'output/'
        image.save(output_file_path)
    
    @staticmethod
    def extract_coordinates(bounds):
        try:
            left_top, right_bottom = bounds.split('][')
            left, top = map(int, left_top.strip('[').split(','))
            right, bottom = map(int, right_bottom.strip(']').split(','))

            # Save coordinates to a JSON file for future use
            coordinates = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
            return coordinates

        except ValueError as e:
            print(f"Error parsing bounds '{bounds}': {e}")
            return None

    @staticmethod
    def draw_bounding_boxes(screenshot_file, elements, output_file):
        image = Image.open(screenshot_file)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        indexed_bounds = {}
        for idx, element in enumerate(elements):
            bounds = element['bounds']
            draw.rectangle([bounds['left'], bounds['top'], bounds['right'], bounds['bottom']], outline='green', width=2)
            text_position = (bounds['left'] + 5, bounds['top'] + 5)
            draw.text(text_position, str(idx + 1), fill='blue', font=font)
            indexed_bounds[idx + 1] = bounds

        image.save(output_file)
        return indexed_bounds

    @staticmethod
    def get_activity_name():
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()
        device = devices[0]
        text = device.shell("dumpsys window displays | grep -E 'mCurrentFocus' ")
        shape = device.shell("wm size")
        width = int(shape.split("x")[0].split(":")[1])
        height = int(shape.split("x")[1])
        
        activity = text.split(".")[-1].split("}")[0]
        with open('cache_storage.json', 'r') as f:
            data = json.load(f)
        output_file = f"dumpimages/{activity}_{len(data) + 1}.png"
        return activity, output_file, len(data) + 1, width, height

   
    @staticmethod
    def capture_screenshot():
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()

        if not devices:
            print("No devices found.")
            return

        device = devices[0]
        keyboard_status = device.shell('dumpsys input_method | grep mInputShown')

        if 'mInputShown=true' in keyboard_status:
            device.shell('input keyevent 4')
            print('Keyboard hidden')

        device.shell('screencap -p /sdcard/screen.png')
        device.pull('/sdcard/screen.png', 'screen.png')
        print('Screenshot captured')

    @staticmethod
    def capture_ui_hierarchy(output_file='ui_hierarchy.xml'):
        try:
            device = u2.connect()
            hierarchy = device.dump_hierarchy(compressed=False)

            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(hierarchy)

            print(f"UI hierarchy saved to {output_file}")

        except Exception as e:
            print(f"Error capturing UI hierarchy: {e}")

    @staticmethod
    def model_response(task, img):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([
            "this is a screenshot of mobile screen with already drawn boxes on elements of the screen along with label numbers inside of boxes. you have to choose the most relevant label number for the given task. \n",
            "\"",
            task,
            "\"",
            "action type and payload if any(if no payload then write none) \n",
            "use this format to output the response ( <label> must be a number) ",
            "target element label: <label> \n",
            "action: <action> \n",
            "for example input, tap, type or write etc \n",
            "payload: <payload> \n",
            "for example if action is type 'text' or write 'text' then payload will be text : none \n",
            img,
        ])

        target_label, action, payload = Utility.parse_response(response.text)
        print("target label is", target_label, "\naction is", action, "\npayload parsed is", payload)
        return target_label, action, payload

    @staticmethod
    def parse_response(response):
        response = response.split("\n")
        target_label = response[0].split(": ")[1].strip()
        action = response[1].split(": ")[1].strip()
        payload = response[2].split(": ")[1].strip()
        return target_label, action, payload
    
    

    @staticmethod
    def execute_task(action, payload, x_cord, y_cord):
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()
        device = devices[0]

        if action == "tap":
            device.shell(f"input tap {x_cord} {y_cord}")
        elif action == "swipe":
            if payload == "down":
                device.shell(f"input touchscreen swipe 530 1500 530 700")
            elif payload == "up":
                device.shell(f"input touchscreen swipe 530 700 530 1500")
        elif action in ["input", "type", "write"]:
            device.shell(f"input tap {x_cord} {y_cord}")
            formatted_payload = payload.replace(" ", "%s")
            device.shell(f"input text {formatted_payload}")
        else:
            print("Invalid action")
            return False

        print("Task executed")
        return True

    @staticmethod
    def bounds_validation(bounds, width, height):
        if bounds['left'] < 0:
            bounds['left'] = 0
        if bounds['top'] < 0:
            bounds['top'] = 0
        if bounds['right'] > width:
            bounds['right'] = width
        if bounds['bottom'] > height:
            bounds['bottom'] = height
        return bounds

    @staticmethod
    def scaleUp(bounds, image_width, image_height, factor=2):
        height = bounds["bottom"] - bounds["top"]
        width = bounds["right"] - bounds["left"]

        new_bounds = {
            "left": int(bounds['left'] - width * factor),
            "top": int(bounds['top'] - height * factor),
            "right": int(bounds['right'] + width * factor),
            "bottom": int(bounds['bottom'] + height * factor)
        }
        new_bounds = Utility.bounds_validation(new_bounds, image_width, image_height)
        return new_bounds

    @staticmethod
    def bounds_to_coords(bounds):
        coords = {
            "x": bounds["left"] + (bounds["right"] - bounds["left"]) / 2,
            "y": bounds["top"] + (bounds["bottom"] - bounds["top"]) / 2,
        }
        return coords

    @staticmethod
    def factor_optimization(area):
        if area < 1000:
            return 25
        elif area < 2000:
            return 20
        elif area < 3000:
            return 15
        elif area < 4000:
            return 10
        elif area < 10000:
            return 4
        elif area < 40000:
            return 1.2
        elif area < 90000:
            return 1
        else:
            return 0.1
        
    @staticmethod
    def bounds_to_action_coords(bounds):
        """
        Converts the bounds of an element to action coordinates by calculating the center point.
        
        :param bounds: A string of bounds in the format "[left,top][right,bottom]".
        :return: A dictionary with 'x' and 'y' coordinates representing the center of the bounds.
        """
        coords = Utility.extract_coordinates(bounds)
        left, right = coords['left'], coords['right']
        top, bottom = coords['top'], coords['bottom']
        x_cord = (left + right) / 2
        y_cord = (top + bottom) / 2
        return {'x': x_cord, 'y': y_cord}
    
    def coordinates_to_bounds(coordinates):
        return '[' + str(coordinates['left']) + ',' + str(coordinates['top']) + '][' + str(coordinates['right']) + ',' + str(coordinates['bottom']) + ']'

    def distance_between_two_bounds(bounds1 , bounds2):
        coordinates1 = Utility.extract_coordinates(bounds1)
        coordinates2 = Utility.extract_coordinates(bounds2)
        x1 = (coordinates1['left'] + coordinates1['right']) / 2
        y1 = (coordinates1['top'] + coordinates1['bottom']) / 2
        x2 = (coordinates2['left'] + coordinates2['right']) / 2
        y2 = (coordinates2['top'] + coordinates2['bottom']) / 2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
    
        
        