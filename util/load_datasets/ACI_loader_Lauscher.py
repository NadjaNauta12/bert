import os
import codecs
from enum import Enum
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

class Type(Enum):
    ENTITY = 1
    RELATION = 2


class Label_Lauscher(Enum):
    BACKGROUND_CLAIM = 1
    OWN_CLAIM = 2
    DATA = 3
    SUPPORTS = 4
    CONTRADICTS = 5
    PARTS_OF_SAME = 6
    SEMANTICALLY_SAME = 7


class Type(Enum):
    ENTITY = 1
    RELATION = 2


class Annotation_Lauscher:
    def __init__(self, id, label, start, end, file, text=""):
        # assign id
        self.id = id

        # assign type
        if id[0] == 'T':
            self.type = Type.ENTITY
        elif id[0] == 'R':
            self.type = Type.RELATION

        # assign label
        if label == "background_claim":
            self.label = Label_Lauscher.BACKGROUND_CLAIM
        elif label == "own_claim":
            self.label = Label_Lauscher.OWN_CLAIM
        elif label == "data":
            self.label = Label_Lauscher.DATA
        elif label == "supports":
            self.label = Label_Lauscher.SUPPORTS
        elif label == "contradicts":
            self.label = Label_Lauscher.CONTRADICTS
        elif label == "parts_of_same":
            self.label = Label_Lauscher.PARTS_OF_SAME
        elif label == "semantically_same":
            self.label = Label_Lauscher.SEMANTICALLY_SAME

        # assign the rest
        if self.type == Type.ENTITY:
            self.start = int(start)
            self.end = int(end)
        else:
            self.start = start
            self.end = end
        self.text = text
        self.file = file

    def to_string(self):
        if self.type == Type.ENTITY:
            type = "Argumentative component annotated: " + str(self.label) + "\n"
            id = "\tId is: " + str(self.id) + "\n"
            range_s = "\tSpan starts on character position: " + str(self.start) + "\n"
            range_e = "\tSpan ends on character position: " + str(self.end) + "\n"
            text = "\tTextual content is: " + str(self.text) + "\n"
            file = "\tIn file: " + str(self.file) + "\n"

            final = type + id + range_s + range_e + text + file + "\n"
        elif self.type == Type.RELATION:
            type = "Argumentative relation annotated: " + str(self.label) + "\n\n"
            range_s = "\tStarting from component: " + str(self.start) + "\n"
            range_e = "\tEnding at component: " + str(self.end) + "\n"
            file = "\tIn file: " + str(self.file) + "\n"

            final = type + range_s + range_e + file + "\n"

        return final

    def as_dict(self):
        return {'File': self.file, 'Annotation_ID': self.id, 'Label': self.label, 'Text': self.text,
                'Type': self.type, 'Start': self.start, 'End': self.end}


def parse_annotations_Lauscher(path):
    stance_occur = 0
    file_counter = 0
    annotations = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.ann' in file:
                file_counter += 1
                with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as ann_file:
                    for line in ann_file:
                        try:
                            id, info, text = line.split("\t")
                            label, start, end = info.split(" ")
                            #file = file[1:3]
                            annotations.append(
                                Annotation_Lauscher(id=id, label=label, start=start, end=end, file=int(file[1:3]), text=text))
                        except Exception as e:
                            # print("\n")
                            # print(e)
                            # print(file)
                            # print(line)
                            # print("\n")
                            if "Stance" in line:
                                stance_occur += 1
                                print("Stance detected", file, line.rstrip(), "Stance#:", stance_occur)
    # print("#stance",stance_occur)
    print("Annotation documents #:", file_counter)
    print("Annotations #:", len(annotations))
    return annotations


def analyze_annotations_Lauscher(annotations):
    print("Number of annotations: " + str(len(annotations)))
    entities = [ann for ann in annotations if ann.type == Type.ENTITY]
    print("\tNumber of entities: " + str(len(entities)))
    claims = [ann for ann in annotations if
              ann.label == Label_Lauscher.BACKGROUND_CLAIM or ann.label == Label_Lauscher.OWN_CLAIM]
    print("\t\tNumber of claims: " + str(len(claims)))
    background_claims = [ann for ann in annotations if ann.label == Label_Lauscher.BACKGROUND_CLAIM]
    print("\t\t\tNumber of background_claims: " + str(len(background_claims)))
    own_claims = [ann for ann in annotations if ann.label == Label_Lauscher.OWN_CLAIM]
    print("\t\t\tNumber of own_claims: " + str(len(own_claims)))
    data = [ann for ann in annotations if ann.label == Label_Lauscher.DATA]
    print("\t\tNumber of data: " + str(len(data)))
    relations = [ann for ann in annotations if ann.type == Type.RELATION]
    print("\tNumber of relations: " + str(len(relations)))
    supports = [ann for ann in annotations if ann.label == Label_Lauscher.SUPPORTS]
    print("\t\tNumber of supports: " + str(len(supports)))
    contradicts = [ann for ann in annotations if ann.label == Label_Lauscher.CONTRADICTS]
    print("\t\tNumber of contradicts: " + str(len(contradicts)))
    parts_of_same = [ann for ann in annotations if ann.label == Label_Lauscher.PARTS_OF_SAME]
    print("\t\tNumber of parts_of_same: " + str(len(parts_of_same)))
    semantically_same = [ann for ann in annotations if ann.label == Label_Lauscher.SEMANTICALLY_SAME]
    print("\t\tNumber of semantically_same: " + str(len(semantically_same)))


if __name__ == "__main__":
    annotations_Lauscher = parse_annotations_Lauscher(
        r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Lauscher\compiled_corpus")
    analyze_annotations_Lauscher(annotations_Lauscher)
    ACI_Annotation_Lauscher = pd.DataFrame([annotation.as_dict() for annotation in annotations_Lauscher])
    print(ACI_Annotation_Lauscher.head())
