# File: target_element_finder.py

import xml.etree.ElementTree as ET
import cv2 as cv
import config
import numpy as np
from utils import Utility 
from computerVision import Computervision
from graphOperations import GraphOperations
from ui_hierarchy_parse import UIHierarchyParser
class TargetElementFinder:
    def __init__(self):
        pass
    UIHierarchyParser.reload_config_module()
    def find_target_element_approach_1(self, cached_xml, current_xml, target_element):
        """
        First approach to find the target element by comparing node similarities.

        :param cached_xml: Path to the cached XML file.
        :param current_xml: Path to the current XML file.
        :param target_element: XML string of the target element.
        :return: Tuple (bool, coordinates) indicating if the element was found and its action coordinates.
        """
        tree1 = ET.parse(cached_xml)
        root1 = tree1.getroot()

        tree2 = ET.parse(current_xml)
        root2 = tree2.getroot()

        target_element = ET.fromstring(target_element)

        if config.DEBUG:
            print('Target element:', target_element.attrib.get('class'), target_element.attrib.get('resource-id'))

        matches_queue = []
        target_element_text = target_element

        for element in root2.iter():
            if len(element) == len(target_element):
                element_text = GraphOperations.extract_text_from_element(element)
                similarity = GraphOperations.calculate_node_similarity(element, target_element)
                
                if config.DEBUG:
                    print(similarity, " ")

                if similarity > 0.9:
                    element_queue = [element]
                    matches_queue.append(element_queue)

        if config.DEBUG:
            print('Matches length:', len(matches_queue))

        if len(matches_queue) == 1:
            return True, Utility.bounds_to_action_coords(matches_queue[0][0].get('bounds'))
        elif len(matches_queue) == 0:
            return False, None

        target_element_parent = target_element

        while len(matches_queue) > 1:
            current_queue = matches_queue[:]
            matches_queue = []
            best_similarity = 0.0

            for element_queue in current_queue:
                element = element_queue[-1]
                similarity = GraphOperations.calculate_node_similarity(element, target_element_parent)
                if similarity > best_similarity:
                    best_similarity = similarity

            if config.DEBUG:
                print('Best similarity:', best_similarity)

            for element_queue in current_queue:
                element = element_queue[-1]
                if (GraphOperations.calculate_node_similarity(element, target_element_parent) >= best_similarity and 
                    best_similarity > 0.98):
                    new_path = element_queue.copy()
                    parent_element = GraphOperations.get_parent(element, root2)

                    if parent_element is not None:
                        new_path.append(parent_element)
                        matches_queue.append(new_path)

            target_element_parent = GraphOperations.get_parent2(target_element_parent, root1)
            
            if config.DEBUG:
                print("Parent of target element:", target_element_parent.tag, target_element_parent.attrib.get('class'), target_element_parent.attrib.get('resource-id'))
                print('Matches queue length:', len(matches_queue))

        if len(matches_queue) == 1:
            bounds = matches_queue[0][0].get('bounds')
            return True, Utility.bounds_to_action_coords(bounds)
        else:
            return False, None

    def find_target_element_approach_2(self, cached_xml, current_xml, target_element, cached_image_path, current_image_path):
        """
        Second approach to find the target element by image comparison and bounding box distances.

        :param cached_xml: Path to the cached XML file.
        :param current_xml: Path to the current XML file.
        :param target_element: XML string of the target element.
        :param cached_image_path: Path to the cached image file.
        :param current_image_path: Path to the current image file.
        :return: Tuple (bool, coordinates) indicating if the element was found and its action coordinates.
        """
        tree1 = ET.parse(cached_xml)
        root1 = tree1.getroot()
        tree2 = ET.parse(current_xml)
        root2 = tree2.getroot()

        target_element = ET.fromstring(target_element)
        matches_queue = self._find_best_matches(cached_image_path, current_image_path, root2, target_element.attrib.get('bounds'), target_element)
        print('Length of matches queue:', len(matches_queue))

        if len(matches_queue) == 1:
            return True, Utility.bounds_to_action_coords(matches_queue[0])
        elif len(matches_queue) == 0:
            return False, None

        elements_queue = []
        for match in matches_queue:
            for element in root2.iter():
                if GraphOperations.verticle_length(element) == GraphOperations.verticle_length(target_element):
                    if Utility.distance_between_two_bounds(element.attrib.get('bounds'), match) < 10:
                        elements = [element]
                        elements_queue.append(elements)

        target_element_parent = GraphOperations.get_parent2(target_element, root1)

        while len(elements_queue) > 1:
            current_queue = elements_queue[:]
            elements_queue = []
            parent_bounds = self._find_best_matches(cached_image_path, current_image_path, root2, target_element_parent.attrib.get('bounds'), target_element_parent)

            for match in parent_bounds:
                for elements in current_queue:
                    element = elements[-1]
                    element_parent = GraphOperations.get_parent2(element, root2)

                    if Utility.distance_between_two_bounds(element_parent.attrib.get('bounds'), match) < 10:
                        elem = elements.copy()
                        elem.append(element_parent)
                        elements_queue.append(elem)

            target_element_parent = GraphOperations.get_parent2(target_element_parent, root1)
            if config.DEBUG:
                print('Length of elements queue:', len(elements_queue))

        if len(elements_queue) == 1:
            return True, Utility.bounds_to_action_coords(elements_queue[0][0].attrib.get('bounds'))
        else:
            return False, None

    def _find_best_matches(self, cached_image_path, current_image_path, root2, target_element_bounds, target_element):
        """
        Private helper function to find best matches using image template matching.

        :param cached_image_path: Path to the cached image.
        :param current_image_path: Path to the current image.
        :param root2: The root node of the current XML.
        :param target_element_bounds: Bounds of the target element.
        :param target_element: The target element.
        :return: List of bounding boxes for best matches.
        """
        cached_image = cv.imread(cached_image_path)
        current_image = cv.imread(current_image_path)
        bounds = Utility.extract_coordinates(target_element_bounds)
        height = bounds['bottom'] - bounds['top']
        width = bounds['right'] - bounds['left']

        cropped_cached_element = cached_image[bounds['top']:bounds['top'] + height, bounds['left']:bounds['left'] + width]
        gray_cached_element = cv.cvtColor(cropped_cached_element, cv.COLOR_BGR2GRAY)
        gray_current_image = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)
        if config.DEBUG:
            cv.imshow('cached element', gray_cached_element)
            cv.waitKey(0)
        result = cv.matchTemplate(gray_current_image, gray_cached_element, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        threshold = 0.95
        loc = np.where(result >= threshold)
        
        if not loc or len(loc) == 0:
            print('Best match not found', result)
            return []

        best_matches = []
        for pt in zip(*loc[::-1]):
            if not any(abs(match[0] - pt[0]) < 10 and abs(match[1] - pt[1]) < 10 for match in best_matches):
                best_matches.append(pt)

        bounds = []
        for match in best_matches:
            if config.DEBUG:
                cv.rectangle(current_image, match, (match[0] + width, match[1] + height), (0, 255, 0), 2)
                cv.imshow('match', current_image)
                cv.waitKey(0)
            image = gray_current_image[match[1]:match[1] + height, match[0]:match[0] + width]
            if config.DEBUG:
                cv.imshow('match', image)
                cv.waitKey(0)
            bound = f'[{match[0]},{match[1]}][{match[0] + width},{match[1] + height}]'
            if config.DEBUG:
                print('Bound:', bound)
            bounds.append(bound)

        return bounds
