import csv
import glob
import os
import re
import sys

if len(sys.argv) == 1:
    folder_path = input("Enter Folder Path of Wiki files: ")
else:
    folder_path = sys.argv[1]

for filename in glob.glob(os.path.join(folder_path, "*.wiki")):
    with open(filename, "r", encoding="utf8") as f:
        contents = f.read()
        fact_store = []
        name = get_facts_from_infobox_film_content()
        if name is None:
            name = get_facts_from_text_contents()
        else:
            get_facts_from_text_contents()
        write_to_tsv(fact_store, folder_path, name)