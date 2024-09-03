import xml.etree.ElementTree as ET
from utils import Utility
class GraphOperations:
    @staticmethod
    def get_parent(node, root):
        """
        Finds the parent of a given node in a tree structure by traversing from the root.
        
        :param node: The target node whose parent is to be found.
        :param root: The root node of the tree.
        :return: The parent node if found; otherwise, the root node or None.
        """
        if node == root or node is None:
            return root
        queue = [(root, None)]  # (current_node, parent_node)
        while queue:
            current_node, parent = queue.pop(0)
            for child in current_node:
                if child == node:
                    return current_node
                queue.append((child, current_node))
        return None

    @staticmethod
    def get_parent2(node, root):
        """
        Finds the parent of a given node in a tree structure by matching the node's attributes 
        (bounds and resource-id) against the child nodes.
        
        :param node: The target node whose parent is to be found.
        :param root: The root node of the tree.
        :return: The parent node if found; otherwise, the root node or None.
        """
        if node == root or node is None:
            return root
        node_bounds = node.attrib.get('bounds')
        node_resource_id = node.attrib.get('resource-id')
        queue = [(root, None)]  # (current_node, parent_node)
        while queue:
            current_node, parent = queue.pop(0)
            for child in current_node:
                if (child.attrib.get('bounds') == node_bounds and 
                    child.attrib.get('resource-id') == node_resource_id and 
                    len(child) == len(node)):
                    return current_node
                queue.append((child, current_node))
        return None

    @staticmethod
    def verticle_length(element):
        """
        Calculates the vertical length (number of nodes) of an element by traversing all its children.
        
        :param element: The root element to start counting from.
        :return: The total count of nodes within the element tree.
        """
        queue = [element]
        length = 0
        while queue:
            current_node = queue.pop(0)
            length += 1
            for child in current_node:
                queue.append(child)
        return length
    
    @staticmethod
    def extract_text_from_element(element):
        """
        Extracts text from an XML element by traversing all child nodes.
        Removes new line characters, extra spaces, and optional stop words.
        
        :param element: The root element to start extraction from.
        :return: Cleaned text string from the element and its children.
        """
        queue = [element]
        text = ""
        while queue:
            current_node = queue.pop(0)
            text_content = current_node.attrib.get('text', "").strip()
            if text_content:
                text += text_content + " "
            for child in current_node:
                queue.append(child)

        text = text.replace('\n', ' ').strip()
        stop_words = set(["a", "the", "and", "is", "in", "at", "of", "on"])
        filtered_text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
        return filtered_text
    
    @staticmethod
    def calculate_node_similarity(node1: ET.Element, node2: ET.Element) -> float:
        """
        Calculates the similarity between two XML nodes based on their attributes, text, and structure.
        
        :param node1: The first XML element.
        :param node2: The second XML element.
        :return: A float representing the similarity score between 0.0 (completely dissimilar) and 1.0 (identical).
        """
        if node1 is None and node2 is None:
            return 1.0
        if node1 is None or node2 is None:
            return 0.0

        similarity_score = 0.0
        total_weight = 0

        # Check if the node is a leaf node
        is_leaf = len(list(node1)) == 0 and len(list(node2)) == 0

        # Compare node class (high importance)
        class_similarity = GraphOperations.compare_attribute(node1, node2, 'class')
        similarity_score += class_similarity * 4
        total_weight += 4

        # Compare resource-id (high importance)
        resource_id_similarity = GraphOperations.compare_attribute(node1, node2, 'resource-id')
        similarity_score += resource_id_similarity * 4
        total_weight += 4

        # Compare text (highest importance for leaf nodes, medium otherwise)
        text_similarity = GraphOperations.compare_attribute(node1, node2, 'text')
        text_weight = 6 if is_leaf else 3
        similarity_score += text_similarity * text_weight
        total_weight += text_weight

        # Compare content-desc (medium importance)
        content_desc_similarity = GraphOperations.compare_attribute(node1, node2, 'content-desc')
        similarity_score += content_desc_similarity * 3
        total_weight += 3

        # Compare other boolean attributes (low importance)
        boolean_attrs = [
            'checkable', 'checked', 'clickable', 'enabled', 'focusable',
            'focused', 'scrollable', 'long-clickable', 'password', 'selected'
        ]
        for attr in boolean_attrs:
            bool_similarity = GraphOperations.compare_attribute(node1, node2, attr)
            similarity_score += bool_similarity
            total_weight += 1

        # Compare bounds (low importance)
        bounds_similarity = GraphOperations.compare_bounds(node1, node2)
        similarity_score += bounds_similarity
        total_weight += 1

        # If not a leaf node, recursively compare children
        if not is_leaf:
            children1 = list(node1)
            children2 = list(node2)
            max_children = max(len(children1), len(children2))

            if max_children > 0:
                child_similarity_sum = sum(
                    GraphOperations.calculate_node_similarity(c1, c2)
                    for c1, c2 in zip(children1 + [None] * (max_children - len(children1)),
                                      children2 + [None] * (max_children - len(children2)))
                )
                avg_child_similarity = child_similarity_sum / max_children
                similarity_score += avg_child_similarity * 5
                total_weight += 5

        return similarity_score / total_weight

    @staticmethod
    def compare_attribute(node1: ET.Element, node2: ET.Element, attr: str) -> float:
        """
        Compares a specific attribute of two XML nodes.
        
        :param node1: The first XML element.
        :param node2: The second XML element.
        :param attr: The attribute to compare.
        :return: A float representing the similarity score (1.0 if identical, 0.0 if different or missing).
        """
        value1 = node1.get(attr)
        value2 = node2.get(attr)

        if value1 is None and value2 is None:
            return 1.0
        if value1 is None or value2 is None:
            return 0.0

        return 1.0 if value1 == value2 else 0.0

    @staticmethod
    def compare_bounds(node1: ET.Element, node2: ET.Element) -> float:
        """
        Compares the bounds attribute of two XML nodes by their relative size.
        
        :param node1: The first XML element.
        :param node2: The second XML element.
        :return: A float representing the similarity score based on bounds size (0.0 to 1.0).
        """
        bounds1 = Utility.extract_coordinates(node1.get('bounds'))
        bounds2 = Utility.extract_coordinates(node2.get('bounds'))

        if not bounds1 or not bounds2:
            return 0.0

        # Compare relative sizes rather than absolute positions
        width1 = bounds1['right'] - bounds1['left']
        height1 = bounds1['bottom'] - bounds1['top']
        width2 = bounds2['right'] - bounds2['left']
        height2 = bounds2['bottom'] - bounds2['top']

        size_similarity = 1 - abs(width1 * height1 - width2 * height2) / max(width1 * height1, width2 * height2)

        return size_similarity