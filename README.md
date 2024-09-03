                                                           
# Automation Task Caching

This repository contains code to cache a given set of instructions for performing automation tasks on Android devices. The project facilitates efficient automation testing by caching instructions and reusing them to save time and resources during repetitive tasks.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: Python should be installed on your machine.
- **Android Studio**: Android Studio must be installed on your machine for Android SDK tools.
- **Device Setup**: Either an Android emulator should be set up, or a real Android device should be configured with Developer Options enabled and USB Debugging turned on.

## How to Start

Follow these steps to set up and run the project:

1. ### **Clone the Repository**:
   ```bash
   git clone https://github.com/harshit3478/drizz_caching 
   cd drizz_caching
   git checkout master
2. ###  **Install Dependencies**:
   ```bash
     pip install -r requirements.txt
3. ### **Connect Your Device**:
     For a real device, connect your phone using a USB cable and ensure USB Debugging is enabled. If using an emulator, ensure it's running.

4. ### **Configure API Key**:
  Create your Gemini API key.
  Add the API key to the config.py file in the project directory.

5. ### **Prepare the Target App**:
  Go to the app you want to test or provide instructions for.

6. ### **Run the Application**:
   ```bash
    python app.py

8.  ###  **Follow On-Screen Instructions**:
   The terminal will guide you through the automation process.

9.  ### **Important Note**:
   Do not remove the empty folder in the project directory, as it is required for caching operations.

## How does it work?

#### -> For the first time, the user will provide the instructions to perform the automation task. The project will cache these instructions for future use.
#### -> If the user wants to perform the same task again, it will be checked if the instructions are already cached. If they are, It will further perform more checks to make sure that it is the same screen and the same element.
#### -> These can be found in caching.py file under the fuction check_if_cached().
#### -> Approaches to check if the element is the same as the one cached is explained below.
#### -> If the instructions are already cached, the project will reuse them to perform the automation task efficiently.


## Approaches Used

The project uses the following approaches to cache and reuse instructions:
***(both of these approaches code can be find in target_elemen_finder.py file)***

### - **Element node matching**:
   ##### ->  In this approach, the target element is extracted from the cached xml file using tree traversal.
   ##### ->  The element is then matched with the current elements on the screen using the calculate node similarity function in the graphOperations.py file.
   ##### ->  If no elements found, function returns false. If one element is found, it returns true with coordinates of the element. If multiple elements are found,  it traverse to the parent of target element and matched elements and again calculate the similarity. 
   ##### ->  Again repeat the process until either no element is found or only one element is found.

### - **Element matching with Image Similarity**:
   ##### ->  In this approach, the target element is extracted from the cached xml file using tree traversal.
   ##### ->  The element is then matched with the current elements on the screen using the find_best_matches function which search for the best match for the target element image in the current screen.
   ##### ->  If no elements found, function returns false. If one element is found, it returns true with coordinates of the element. If multiple elements are found, it traverse to the parent of target element and matched elements and again calculate the similarity.
   ##### ->  Again repeat the process until either no element is found or only one element is found.

   



