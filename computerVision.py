import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from skimage.metrics import structural_similarity as ssim
from utils import Utility

class Computervision:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def similarity_score(self, text1, text2):
        """Calculates the cosine similarity between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The cosine similarity score.
        """

        vectors = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors)
        return float(similarity[0][1])

    def image_match(self, image_path1, image_path2):
        """Calculates the SSIM between two images.

        Args:
            image_path1 (str): The path to the first image.
            image_path2 (str): The path to the second image.

        Returns:
            float: The SSIM score.
        """

        start = time.time()
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(gray1, gray2, full=True)
        end = time.time()
        print("time taken for image match:", end - start)
        return float(score)

    def cropped_image_match(self, image_path1, image_path2, bounds):
        """Calculates the SSIM between two cropped images.

        Args:
            image_path1 (str): The path to the first image.
            image_path2 (str): The path to the second image.
            bounds (dict): The bounding box of the cropped region.

        Returns:
            tuple: A tuple containing the SSIM score and a dictionary of action coordinates.
        """

        start = time.time()
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        cropped_image1 = image1[bounds["top"]:bounds["bottom"], bounds["left"]:bounds["right"]]
        gray_cropped1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_image2, gray_cropped1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        h, w = gray_cropped1.shape
        best_match = gray_image2[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

        score, _ = ssim(gray_cropped1, best_match, full=True)
        end = time.time()
        print("Time taken for cropped image match:", end - start)

        action_coords = {
            "left": max_loc[0],
            "top": max_loc[1],
            "right": max_loc[0] + w,
            "bottom": max_loc[1] + h
        }

        if float(score) > 0.9:
            return action_coords, True
        return float(score), False

    def match_vicinity(self, image_path1, image_path2, bounds, factor=1):
        """Matches the vicinity of a target element in an image.

        Args:
            image_path1 (str): The path to the first image.
            image_path2 (str): The path to the second image.
            bounds (dict): The bounding box of the target element.
            factor (float, optional): The scaling factor for the vicinity. Defaults to 1.

        Returns:
            tuple: A tuple containing the SSIM score and a dictionary of action coordinates.
        """

        start = time.time()
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        height = bounds["bottom"] - bounds["top"]
        width = bounds["right"] - bounds["left"]
        factor = Utility.factor_optimization(height * width)
        height = int(height * factor)

        cropped_image1 = image1[bounds["top"] - height:bounds["bottom"] + height, 0:-1]

        gray_cropped1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_image2, gray_cropped1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        h, w = gray_cropped1.shape
        best_match = gray_image2[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

        score, _ = ssim(gray_cropped1, best_match, full=True)
        end = time.time()
        print("Time taken for cropped image match:", end - start)

        action_coords = {
            "left": bounds['left'],
            "top": max_loc[1] + height,
            "right": bounds['right'],
            "bottom": max_loc[1] + height + (height / factor)
        }

        coords = {
            "x": bounds['left'] + (bounds['right'] - bounds['left']) / 2,
            "y": max_loc[1] + height + (height / factor) / 2
        }

        if float(score) > 0.9:
            return coords, True
        return float(score), False

    def match_vicinity2(self, image_path1, image_path2, bounds, factor=1):
        """Matches the vicinity of a target element in an image, considering horizontal swipes.

        Args:
            image_path1 (str): The path to the first image.
            image_path2 (str): The path to the second image.
            bounds (dict): The bounding box of the target element.
            factor (float, optional): The scaling factor for the vicinity. Defaults to 1.

        Returns:
            tuple: A tuple containing the SSIM score and a dictionary of action coordinates.
        """

        start = time.time()
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        height = bounds["bottom"] - bounds["top"]
        width = bounds["right"] - bounds["left"]
        factor = Utility.factor_optimization(height * width)
        height = int(height * factor)

        cropped_image1 = image1[bounds["top"] - height:bounds["bottom"] + height, 0:-1]

        gray_cropped1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_image2, gray_cropped1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        h, w = gray_cropped1.shape
        best_match = gray_image2[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

        score, _ = ssim(gray_cropped1, best_match, full=True)

        cropped_image2 = image1[bounds["top"]:bounds["bottom"], bounds['left']:bounds['right']]
        gray_cropped2 = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2GRAY)

        new_result = cv2.matchTemplate(best_match, gray_cropped2, cv2.TM_CCOEFF_NORMED)
        _, new_max_val, _, new_max_loc = cv2.minMaxLoc(new_result)
        h, w = gray_cropped2.shape
        new_best_match = best_match[new_max_loc[1]:new_max_loc[1] + h, new_max_loc[0]:new_max_loc[0] + w]

        new_score, _ = ssim(gray_cropped2, new_best_match, full=True)

        end = time.time()
        print("Time taken for cropped image match:", end - start)

        action_coords = {
            "left": new_max_loc[0],
            "top": max_loc[1] + height,
            "right": new_max_loc[0] + w,
            "bottom": max_loc[1] + height + (height / factor)
        }

        coords = {
            "x": new_max_loc[0] + w / 2,
            "y": max_loc[1] + height + (height / factor) / 2
        }

        return score, coords

    def match_vicinity3(self, image_path1, image_path2, bounds, factor=2):
        """Matches the vicinity of a target element in an image, using scaling up.

        Args:
            image_path1 (str): The path to the first image.
            image_path2 (str): The path to the second image.
            bounds (dict): The bounding box of the target element.
            factor (float, optional): The scaling factor for the vicinity. Defaults to 2.

        Returns:
            tuple: A tuple containing the SSIM score and a dictionary of action coordinates.
        """

        start = time.time()
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        height = bounds["bottom"] - bounds["top"]
        width = bounds["right"] - bounds["left"]
        factor = Utility.factor_optimization(height * width)
        new_bounds = Utility.scaleUp(bounds, image1.shape[1], image1.shape[0], factor)

        cropped_image1 = image1[new_bounds["top"]:new_bounds["bottom"], new_bounds['left']:new_bounds['right']]

        gray_cropped1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_image2, gray_cropped1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        h, w = gray_cropped1.shape
        best_match = gray_image2[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

        score, _ = ssim(gray_cropped1, best_match, full=True)

        cropped_image2 = image1[bounds["top"]:bounds["bottom"], bounds['left']:bounds['right']]
        gray_cropped2 = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2GRAY)

        new_result = cv2.matchTemplate(best_match, gray_cropped2, cv2.TM_CCOEFF_NORMED)
        _, new_max_val, _, new_max_loc = cv2.minMaxLoc(new_result)
        h, w = gray_cropped2.shape
        new_best_match = best_match[new_max_loc[1]:new_max_loc[1] + h, new_max_loc[0]:new_max_loc[0] + w]

        new_score, _ = ssim(gray_cropped2, new_best_match, full=True)

        end = time.time()
        print("Time taken for cropped image match:", end - start)

        action_coords = {
            "left": max_loc[0] + new_max_loc[0],
            "top": max_loc[1] + new_max_loc[1],
            "right": max_loc[0] + new_max_loc[0] + w,
            "bottom": max_loc[1] + new_max_loc[1] + h
        }

        coords = Utility.bounds_to_coords(action_coords)

        return score