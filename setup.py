import os


# if dumpimages folder does not exist, create it
if not os.path.exists("dumpimages"):
    os.makedirs("dumpimages")
    
# delete files in dump images folder
for file in os.listdir('dumpimages'):
    os.remove(f'dumpimages/{file}')
    
if not os.path.exists("images"):
    os.makedirs("images")

for file in os.listdir('images'):
    os.remove(f'images/{file}')
    
if not os.path.exists("output"):
    os.makedirs("output")

for file in os.listdir('output'):
    os.remove(f'output/{file}')

if not os.path.exists("xml_files"):
    os.makedirs("xml_files")

for file in os.listdir('xml_files'):
    os.remove(f'xml_files/{file}')



os.remove('indexed_bounds.json')
os.remove('screen.png')
os.remove('ui_hierarchy.xml')
os.remove('cache_storage.json')

with open("cache_storage.json", mode='w', encoding='utf-8') as feedsjson:
    feedsjson.write("[]")
