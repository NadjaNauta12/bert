import os
import codecs
from enum import Enum
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Type(Enum):
    ENTITY = 1
    RELATION = 2


class Label_Habernal(Enum):
    """
    [entities]
        MajorClaim
        Claim
        Premise
    [relations]
        supports Arg1:Premise|Claim,Arg2:Claim|MajorClaim|Premise
        attacks  Arg1:Premise|Claim,Arg2:Claim|MajorClaim|Premise
    [attributes]
        Stance   	Arg:Claim, Value:For|Against
    [events]
    """
    MAJOR_CLAIM = 1
    CLAIM = 2
    PREMISE = 3
    SUPPORTS = 4
    ATTACKS = 3


class Annotation_Habernal:
    def __init__(self, id, label, start, end, file, text=""):
        # assign id
        self.id = id

        # assign type
        if id[0] == 'T':
            self.type = Type.ENTITY
        elif id[0] == 'R':
            self.type = Type.RELATION

        label = label.lower()
        # assign label
        if label == "majorclaim":
            self.label = Label_Habernal.MAJOR_CLAIM
        elif label == "claim":
            self.label = Label_Habernal.CLAIM
        elif label == "premise":
            self.label = Label_Habernal.PREMISE
        elif label == "supports":
            self.label = Label_Habernal.SUPPORTS
        elif label == "attacks":
            self.label = Label_Habernal.ATTACKS
        else:
            self.label = "not defined"
            # dict_label[label] = dict_label.get(label, 0) + 1

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


def analyze_annotations_Habernal(annotations):
    print("Number of annotations: " + str(len(annotations)))
    for a in annotations:
        for b in a:
            print(b)

    entities = [ann for ann in annotations if ann.type == Type.ENTITY]
    print("\tNumber of entities: " + str(len(entities)))
    claims = [ann for ann in annotations if
              ann.label == Label_Habernal.MAJOR_CLAIM or ann.label == Label_Habernal.CLAIM]
    print("\t\tNumber of claims: " + str(len(claims)))
    major_claims = [ann for ann in annotations if ann.label == Label_Habernal.MAJOR_CLAIM]
    print("\t\t\tNumber of major_claims: " + str(len(major_claims)))
    claims = [ann for ann in annotations if ann.label == Label_Habernal.CLAIM]
    print("\t\t\tNumber of claims: " + str(len(claims)))
    premises = [ann for ann in annotations if ann.label == Label_Habernal.PREMISE]
    print("\t\t\tNumber of premises: " + str(len(premises)))
    # data = [ann for ann in annotations if ann.label == Label_Habernal.DATA]
    # print("\t\tNumber of data: " + str(len(data)))
    relations = [ann for ann in annotations if ann.type == Type.RELATION]
    print("\tNumber of relations: " + str(len(relations)))
    supports = [ann for ann in annotations if ann.label == Label_Habernal.SUPPORTS]
    print("\t\tNumber of supports: " + str(len(supports)))
    attacks = [ann for ann in annotations if ann.label == Label_Habernal.ATTACKS]
    print("\t\tNumber of attacks: " + str(len(attacks)))


def parse_annotations_Habernal(
        path=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final"):
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
                            try:
                                id, info, text = line.split("\t")
                            except ValueError as valerr:
                                if "Premise" in line:
                                    id, info, text, empty = line.split("\t")
                                elif "Stance" in line:
                                    stance_occur += 1
                                    # print("Stance detected", file, line.rstrip(), "Stance#:", stance_occur, "ignored")
                                else:
                                    print("something is fischy", valerr)

                            label, start, end = info.split(" ")

                            annotations.append( Annotation_Habernal(id=id, label=label, start=start, end=end, file=int(file[5:7]), text=text))
                        except Exception as e:
                            # prrint("\n")
                            print(e)
                            # print(file)
                            # print(line)
                            print("Something is not right...\n")

    # print("#stance",stance_occur)
    print("Annotation documents #:", file_counter)
    print("Annotations #:", len(annotations))
    # print("Found labels + counts:", dict_label)
    df = pd.DataFrame([annotation.as_dict() for annotation in annotations])
    #df = pd.get_dummies(df["Label"])
    df.Label = df["Label"].astype(str)
    #total_rows['ColumnID'].astype(str)
    LE = LabelEncoder()
    df['target_label'] = LE.fit_transform(df['Label'])
    return df  # annotations


annotations_Habernal = parse_annotations_Habernal(
   r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final")
analyze_annotations_Habernal(annotations_Habernal)
ACI_Annotation_Habernal = pd.DataFrame([annotation.as_dict() for annotation in annotations_Habernal])
print(ACI_Annotation_Habernal.head())
