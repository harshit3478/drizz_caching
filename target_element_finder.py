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

        # target_element_text = target_element
        matches_queue = []

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
        # if config.DEBUG:
        #     cv.imshow('cached element', gray_cached_element)
        #     cv.waitKey(0)
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
            # if config.DEBUG:
            #     cv.rectangle(current_image, match, (match[0] + width, match[1] + height), (0, 255, 0), 2)
            #     cv.imshow('match', current_image)
            #     cv.waitKey(0)
            image = gray_current_image[match[1]:match[1] + height, match[0]:match[0] + width]
            # if config.DEBUG:
            #     cv.imshow('match', image)
            #     cv.waitKey(0)
            bound = f'[{match[0]},{match[1]}][{match[0] + width},{match[1] + height}]'
            if config.DEBUG:
                print('Bound:', bound)
            bounds.append(bound)

        return bounds

    def find_best_match_elements(self, cached_image_path, current_image_path, root2, target_element_bounds, target_element):
        """
        Find the best matching elements using image template matching.
        """
        
    
    def find_target_element_approach3(self, target_element_string, cached_image , current_image , cached_xml , current_xml):
        """
        Third approach to find the target element by comparing the image of the target element with the current image.

        :param target_element: XML string of the target element.
        :param cached_image: Path to the cached image.
        :param current_image: Path to the current image.
        :param cached_xml: Path to the cached XML file.
        :param current_xml: Path to the current XML file.
        :return: Tuple (bool, coordinates) indicating if the element was found and its action coordinates.
        """
        CV = Computervision()
        print('Cached xml path:', cached_xml)
        tree1 = ET.parse(cached_xml)
        root1 = tree1.getroot()
        tree2 = ET.parse(current_xml)
        root2 = tree2.getroot()
        
        target_element = ET.fromstring(target_element_string)
        # find the exact element address for the target element in cached xml
        index = 100
        for element in root1.iter():
            # print(element)
            if len(element) == len(target_element) and element.attrib.get('class') == target_element.attrib.get('class') and element.attrib.get('resource-id') == target_element.attrib.get('resource-id'):
                if element.attrib.get('bounds') is not None:
            
                    # node_similarity = GraphOperations.calculate_node_similarity(element, target_element)
                    bounds_similarity = element.attrib.get('bounds') == target_element.attrib.get('bounds')
                    # image_similarity = CV.match_two_cropped_images(cached_image, cached_image , Utility.extract_coordinates(element.attrib.get('bounds')), Utility.extract_coordinates(target_element.attrib.get('bounds')), index  )
                    if  bounds_similarity :
                        target_element = element
                        print('Target element found in cached xml')
                        break
                    index += 1
        
        target_element_ancestors = self.get_ancestor_elements(target_element, root1)
        # for node in target_element_ancestors:
        #     if config.DEBUG:
        #         print('Ancestor:', node.tag, node.attrib.get('class'), node.attrib.get('resource-id'))
     
        
        matches_queue = []
        depth_of_target_element = GraphOperations.verticle_depth(target_element, root1)

        for element in root2.iter():
            depth_of_element = GraphOperations.verticle_depth(element, root2)  
            if depth_of_element == depth_of_target_element:
                similarity = GraphOperations.calculate_node_similarity(element, target_element)
                if similarity > 0.95:
                    # element_queue = [element]
                    matches_queue.append(element)
                    
        if config.DEBUG:
            print('Length of matches queue:', len(matches_queue))
        if len(matches_queue) == 1:
            is_found, action_coords = self.find_target_element_approach_2(cached_xml, current_xml, target_element_string, cached_image, current_image)
            return is_found, action_coords
        if len(matches_queue) == 0:
            return False, None
        can_element_ancestors_queue = []
        for match in matches_queue:
            can_element_ancestors = self.get_ancestor_elements(match, root2)
            can_element_ancestors_queue.append(can_element_ancestors)
            if config.DEBUG:
                print('Ancestors of candidate element: ',can_element_ancestors[0],  len(can_element_ancestors))
        
        #if more than one match than find common ancestor and it's verticle depth
        common_ancestor, depth_of_common_ancestor = self.find_common_ancestor(can_element_ancestors_queue[0], can_element_ancestors_queue[1], root2)
        # print('depth of common ancestor:', depth_of_common_ancestor)
        if config.DEBUG:
            print('Length of can_element_ancestors_queue:', len(can_element_ancestors_queue))
            
        is_found, action_coords = self.check_ancestors(target_element_ancestors, can_element_ancestors_queue, cached_image, current_image , depth_of_common_ancestor)
        
        if is_found:
            return True, action_coords
        else:
            if config.DEBUG:
                print('Target element not found')
            return False, None
                
        
    def get_ancestor_elements( self , element, root):
        """
        Get the ancestor elements of the given element.

        :param element: The XML element.
        :param root: The root of the XML tree.
        :return: A list of ancestor elements.
        """
        #recursively find the ancestors of the element
        ancestors = [element]
        parent = element
        while parent is not None and parent != root:
            parent = GraphOperations.get_parent(parent, root)
            if parent is not None:
                ancestors.append(parent)
        return ancestors
    
    def check_ancestors(self, target_element_ancestors, can_element_ancestors_queue, cached_image, current_image, depth_of_common_ancestor):
        """
        Check if the ancestors of the target element are present in the current XML.

        :param target_element_ancestors: Ancestor elements of the target element.
        :param can_element_ancestors: Ancestor elements of the candidate element.
        :return: True if the ancestors are present; False otherwise.
        """
        CV = Computervision()
        
        cached_image = cv.imread(cached_image)
        current_image = cv.imread(current_image)
        cv.imwrite('output/image1.png', cached_image)
        cv.imwrite('output/image2.png', current_image)
        matched_paths = []
        for can_ancestors in can_element_ancestors_queue :
            n = len(can_ancestors)
            flag = True
            for i in range(n-(depth_of_common_ancestor+1)):
                threshold = 0.9
                bounds1 = Utility.extract_coordinates(target_element_ancestors[i].attrib.get('bounds'))
                bounds2 = Utility.extract_coordinates(can_ancestors[i].attrib.get('bounds'))
                #resize the image to the size of the element for the smaller image
                target_element_width  = bounds1['right'] - bounds1['left']
                target_element_height = bounds1['bottom'] - bounds1['top']
                current_element_width  = bounds2['right'] - bounds2['left']
                current_element_height = bounds2['bottom'] - bounds2['top']

                if target_element_width*target_element_height > current_element_width*current_element_height:
                    bounds1['right'] = bounds1['left'] + current_element_width
                    bounds1['bottom'] = bounds1['top'] + current_element_height
                else:
                    bounds2['right'] = bounds2['left'] + target_element_width
                    bounds2['bottom'] = bounds2['top'] + target_element_height
                
                
                
                #copy cached image and current image in output/image1.png and output/image2.png respectively first convert to numpy array
                if cached_image is None or current_image is None:
                    raise FileNotFoundError(f"Cannot open or read the file: {cached_image}. Please check the path and file integrity.")
                score = CV.match_two_cropped_images('output/image1.png', 'output/image2.png', bounds1, bounds2, i, True)
                # print('score of matching elements in check ancestors:', score)
                if score < threshold:
                    flag = False
                    break
            if flag:
                matched_paths.append(can_ancestors)
            
        if len(matched_paths) == 1:
            action_coords = Utility.bounds_to_action_coords(matched_paths[0][0].attrib.get('bounds'))
            return True, action_coords
        elif len(matched_paths) == 0:
            print('No match found')
            return False, None
        else :
            print('more than one match found ')
                
        return False, None
            
    def find_common_ancestor(self ,ancestors1 , ancestors2 , root):
        """
        Find the common ancestor of two elements.

        :param element1: The first element.
        :param element2: The second element.
        :param root: The root of the XML tree.
        :return: The common ancestor element.
        """
        common_ancestor = None
        for ancestor1 in ancestors1:
            for ancestor2 in ancestors2:
                if ancestor1 == ancestor2:
                    common_ancestor = ancestor1
                    break
            if common_ancestor is not None:
                break
        return common_ancestor, GraphOperations.verticle_depth(common_ancestor, root)
    