import json
import os
from computerVision import Computervision
import config
from target_element_finder import TargetElementFinder
from utils import Utility
from ui_hierarchy_parse import UIHierarchyParser
# creat a class Cache

class Cache:
    CV = Computervision()
    TEF = TargetElementFinder()
    updates = {}
    
    @staticmethod
    def cache_task(task, imagePath, element_coords, activityName, action, payload, xml_path):
        entry = {
            'instruction': task,
            'activity_name': activityName,
            'image': imagePath,
            'element_bounds': element_coords,
            'action': action,
            'payload': payload,
            'xml_path': xml_path
        }

        if os.path.exists("cache_storage.json"):
            with open("cache_storage.json", mode='r', encoding='utf-8') as feedsjson:
                feeds = json.load(feedsjson)
        else:
            feeds = []

        feeds.append(entry)

        with open("cache_storage.json", mode='w', encoding='utf-8') as feedsjson:
            json.dump(feeds, feedsjson, indent=4) 
            
    @staticmethod
    def check_if_cached(task, activity_name, imagePath):
        """Checks if a task is cached and executes it if found.

        Args:
            task (str): The task description.
            activity_name (str): The activity name.
            imagePath (str): The path to the image.

        Returns:
            bool: True if the task was found in the cache and executed, False otherwise.
        """
        CV = Cache.CV
        TEF = Cache.TEF
        with open("cache_storage.json", mode='r', encoding='utf-8') as feedsjson:
            feeds = json.load(feedsjson)

        for document in feeds:
            # Calculate similarity scores
            task_similarity = CV.similarity_score(  task , document['instruction'])
            if task_similarity < 0.60:
                continue
            
            activity_similarity = CV.similarity_score( activity_name, document['activity_name'])
            if activity_similarity < 0.99:
                continue
            # image_similarity = CV.image_match( imagePath, document['image'])
            cropped_image_similarity = CV.cropped_image_match(imagePath , document['image'], document['element_bounds'])
            # if cropped_image_similarity < 0.99:
            #     continue
            # if image_similarity < 0.50:
            #     continue
            
            if config.DEBUG:
                print('task similarity :' , task_similarity , ' for cached task :', document['instruction'])
                print('activity similarity :' , activity_similarity , ' for cached activity :', document['activity_name'])
                # print('image similarity :' , image_similarity, ' for cached image :', document['image'])
                print('cropped image similarity :' , cropped_image_similarity, ' for cached image :', document['image'])
           
            # Check for matching target element 
            #approach 1
            # is_match , action_coords = TEF.find_target_element_approach_1(
            #     document['xml_path'],
            #     'ui_hierarchy.xml',
            #     document['element_bounds']['node']
            # )
            #approach 2                
            # is_match, action_coords = TEF.find_target_element_approach_2(
            #     document['xml_path'],
            #     'ui_hierarchy.xml',
            #     document['element_bounds']['node'],
            #     document['image'],
            #     'screen.png'
            # )
            
            #approach 3
            is_match, action_coords = TEF.find_target_element_approach3(
                document['element_bounds']['node'],
                document['image'],
                'screen.png',
                document['xml_path'],
                'ui_hierarchy.xml'
                )

            if is_match:
                print("Task found in cache and executed successfully")
                Utility.execute_task(document['action'], document['payload'],
                            str(action_coords['x']), str(action_coords['y']))
                return True

            # If any similarity is below threshold, continue to next document

        print("Task not found in cache")
        return False