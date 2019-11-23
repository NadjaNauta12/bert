import os
import codecs
import pandas as pd


class Essay():
    def __init__(self, essay,  text=""):
        # assign id
        self.essay = essay
        self.text = text

    def as_dict(self):
        return {'file': self.essay, 'text': self.text}


def parse_essays(path):
    essays = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file:
                with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as essay_file:
                    content = essay_file.read()
                    essays.append(Essay(int(file[5:7]), content))

    return essays


if __name__ == "__main__":
    essay_list = parse_essays(
        r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final")
    df_essay = pd.DataFrame.from_records([essay.as_dict() for essay in essay_list])
    print(df_essay.head())