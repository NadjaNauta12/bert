import os
import codecs

def parse_essays(path):
    essays = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file:
                with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as essay_file:
                    essays.append(Essay(file, essay_file))

    return essays

class Essay():
    def __init__(self, file,  text=""):
        # assign id
        self.file = file
        self.text = text


essays = parse_essays(r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final")