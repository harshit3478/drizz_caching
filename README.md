                                                           
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
