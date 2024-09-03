import subprocess
from utils import Utility
def run_script():
    scripts = ['ui_hierarchy_parse.py' , 'main.py']
    flag = True
    while(flag):
        Utility.capture_screenshot()
        Utility.capture_ui_hierarchy()
        
        for script in scripts:
            subprocess.run(['python', script])
        text = input("do you want to continue? type anything to discontinue : ")
        if(text):
            flag = False
            
if __name__ == "__main__":
    run_script()