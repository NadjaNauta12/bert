import os
import codecs
import pandas as pd

def parse_essays(path):
    essays = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file:
                with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as essay_file:
                    content = essay_file.read()
                    essays.append(Essay(int(file[5:7]), content))

    return essays

class Essay():
    def __init__(self, essay,  text=""):
        # assign id
        self.essay = essay
        self.text = text

    def as_dict(self):
        return {'file': self.essay, 'text': self.text}




