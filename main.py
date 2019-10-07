
import glob
import os

import sys

if len(sys.argv) == 1:
    folder_path = input("Enter Folder Path of Training files: ")
else:
    folder_path = sys.argv[1]

for filename in glob.glob(os.path.join(folder_path, "*.tra")):
    f = open (filename, "r")
    contents = f.read()
    print(contents)