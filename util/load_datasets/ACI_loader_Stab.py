import os
import codecs
import pandas as pd
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

'''Simple class for representing the desired output'''


class BRAT_Token:
    def __init__(self, token, start, end, token_label=None, sentence_label=None, is_end_of_sentence=False, file=None):
        self.token = token
        self.start = start
        self.end = end
        if token_label is not None:
            self.token_label = token_label
        else:
            self.token_label = Token_Label.OUTSIDE
        self.sentence_label = sentence_label
        self.matched = False
        self.is_end_of_sentence = is_end_of_sentence
        self.file = file

    def get_token_annotation(self):
        return str(self.token) + "\t" + str(self.token_label) + "\n"


class Type(Enum):
    ENTITY = 1
    RELATION = 2


'''Enum for representing our argument labels'''


class Token_Label(Enum):
    BEGIN_MAJOR_CLAIM = 1
    INSIDE_MAJOR_CLAIM = 2
    BEGIN_CLAIM = 3
    INSIDE_CLAIM = 4
    BEGIN_PREMISE = 5
    INSIDE_PREMISE = 6
    OUTSIDE = 7


class Essay:
    def __init__(self, essay, text=""):
        # assign id
        self.essay = essay
        self.text = text
        self.sentences = [token for token in span_tokenize_sentences(text)]
        self.tokens = [token for token in span_tokenize(text)]

    def as_dict(self):
        return {'file': self.essay, 'text': self.text}


class Label_Stab(Enum):
    """
    [entities] >> MajorClaim    Claim   Premise
    [relations] >>  supports Arg1:Premise|Claim,Arg2:Claim|MajorClaim|Premise
                >>  attacks  Arg1:Premise|Claim,Arg2:Claim|MajorClaim|Premise
    NOT CONSIDERED: [attributes] >>  Stance   	Arg:Claim, Value:For|Against
                    [events]
    """
    MAJOR_CLAIM = 1
    CLAIM = 2
    PREMISE = 3
    SUPPORTS = 4
    ATTACKS = 5


class Annotation_Stab:
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
            self.label = Label_Stab.MAJOR_CLAIM
        elif label == "claim":
            self.label = Label_Stab.CLAIM
        elif label == "premise":
            self.label = Label_Stab.PREMISE
        elif label == "supports":
            self.label = Label_Stab.SUPPORTS
        elif label == "attacks":
            self.label = Label_Stab.ATTACKS
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


def parse_essays(path):
    essays = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file:
                with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as essay_file:
                    content = essay_file.read()
                    # essays.append(Essay(int(file[5:7]), content))
                    essays.append(Essay(file, content))

    return essays


def parse_annotations_Stab(path):
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
                                if id[0] == 'R':
                                    continue
                            except ValueError as valerr:
                                if "Premise" in line:
                                    id, info, text, empty = line.split("\t")
                                elif "Stance" in line:
                                    stance_occur += 1
                                    continue
                                    # print("Stance detected", file, line.rstrip(), "Stance#:", stance_occur, "ignored")
                                else:
                                    print("something is fishy", valerr)

                            label, start, end = info.split(' ')

                            annotations.append(
                                Annotation_Stab(id=id, label=label, start=start, end=end, file=int(file[5:7]),
                                                    text=text))

                        except Exception as e:
                            # print("\n")
                            print(e)
                            # print(file)
                            print(line)
                            # print("Something is not right...\n")
            # print("#stance",stance_occur)
    print("Annotation documents #:", file_counter)
    print("Annotations #:", len(annotations))
    # print("Found labels + counts:", dict_label)
    analyze_annotations_Stab(annotations)
    # return annotations
    df = pd.DataFrame([annotation.as_dict() for annotation in annotations])
    # df = pd.get_dummies(df["Label"])
    df.Label = df["Label"].astype(str)
    # total_rows['ColumnID'].astype(str)
    LE = LabelEncoder()
    df['target_label'] = LE.fit_transform(df['Label'])
    return df  # annotations


def parse_annotations_Stab_perFile(path):
    stance_occur = 0
    file_counter = 0
    annotations_perFile = []
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
                                if id[0] == 'R':
                                    continue
                            except ValueError as valerr:
                                if "Premise" in line:
                                    id, info, text, empty = line.split("\t")
                                elif "Stance" in line:
                                    stance_occur += 1
                                    continue
                                    # print("Stance detected", file, line.rstrip(), "Stance#:", stance_occur, "ignored")
                                elif "Claim" in line:  # !!  one line has a \t in the description text'
                                    id, info, text, text2 = line.split("\t")
                                    text = text + " " + text2
                                else:
                                    print("something is fishy", valerr)

                            label, start, end = info.split(' ')

                            annotations.append(
                                Annotation_Stab(id=id, label=label, start=start, end=end, file=file, text=text))

                        except Exception as e:
                            # print("\n")
                            print(e)
                            # print(file)
                            print(line)
                            # print("Something is not right...\n")
                # annotations_perFile.append(annotations)
    return annotations


def analyze_annotations_Stab(annotations):
    print("Number of annotations: " + str(len(annotations)))
    entities = [ann for ann in annotations if ann.type == Type.ENTITY]
    print("\tNumber of entities: " + str(len(entities)))
    claims = [ann for ann in annotations if
              ann.label == Label_Stab.MAJOR_CLAIM or ann.label == Label_Stab.CLAIM]
    print("\t\tNumber of claims: " + str(len(claims)))
    major_claims = [ann for ann in annotations if ann.label == Label_Stab.MAJOR_CLAIM]
    print("\t\t\tNumber of major_claims: " + str(len(major_claims)))
    claims = [ann for ann in annotations if ann.label == Label_Stab.CLAIM]
    print("\t\t\tNumber of claims: " + str(len(claims)))
    premises = [ann for ann in annotations if ann.label == Label_Stab.PREMISE]
    print("\t\t\tNumber of premises: " + str(len(premises)))
    # data = [ann for ann in annotations if ann.label == Label_Stab.DATA]
    # print("\t\tNumber of data: " + str(len(data)))
    relations = [ann for ann in annotations if ann.type == Type.RELATION]
    print("\tNumber of relations: " + str(len(relations)))
    supports = [ann for ann in annotations if ann.label == Label_Stab.SUPPORTS]
    print("\t\tNumber of supports: " + str(len(supports)))
    attacks = [ann for ann in annotations if ann.label == Label_Stab.ATTACKS]
    print("\t\tNumber of attacks: " + str(len(attacks)))


'''Takes the BRAT annotations and the corresponding text files and joins them accordingly'''


def join_argument_annotations_and_texts(texts, annotations):
    for doc in texts:
        for ann in annotations:
            if doc.essay.split('.txt')[0] == ann.file.split('.ann')[0]:
                if ann.type == Type.ENTITY:
                    for token in doc.tokens:
                        try:
                            if token.start == ann.start and ann.label == Label_Stab.CLAIM:
                                token.token_label = Token_Label.BEGIN_CLAIM
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == Label_Stab.CLAIM:
                                token.token_label = Token_Label.INSIDE_CLAIM
                            elif token.start == ann.start and ann.label == Label_Stab.MAJOR_CLAIM:
                                token.token_label = Token_Label.BEGIN_MAJOR_CLAIM
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == Label_Stab.MAJOR_CLAIM:
                                token.token_label = Token_Label.INSIDE_MAJOR_CLAIM
                            elif token.start == ann.start and ann.label == Label_Stab.PREMISE:
                                token.token_label = Token_Label.BEGIN_PREMISE
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == Label_Stab.PREMISE:
                                token.token_label = Token_Label.INSIDE_PREMISE
                        except Exception as e:
                            print(e)


'''Function that should not only tokenize the text but also return the character positions'''


def span_tokenize(text):
    tokens = word_tokenize(text)
    offset = 0
    offset_new = 0
    for token in tokens:
        if token == "\'\'" or token == "``":  # windows has problems with these symbols that are ""
            token = '"'
        offset_new = text.find(token, offset)
        if offset_new < 0:
            print("Problem occured with token " + token)
        else:
            offset = offset_new
            yield BRAT_Token(token, offset, offset + len(token))
            offset += len(token)


'''Function that should not only tokenize the text but also return the character positions for sentences'''


def span_tokenize_sentences(text):
    tokens = sent_tokenize(text)
    offset = 0
    offset_new = 0
    for token in tokens:
        offset_new = text.find(token, offset)
        if offset_new < 0:
            print("Problem occured with token " + token)
        else:
            offset = offset_new
            yield BRAT_Token(token, offset, offset + len(token))
            offset += len(token)


def write_annotated_essays(essay_list):
    path = ""   # TODO PATH  data\Argument_Component_Identification_Stab\compiled_corpus
    for essay_ele in essay_list:
        path2 = path + "\\" + essay_ele.essay[:len(essay_ele.essay) - 4]
        file = open(path + "\\" + essay_ele.essay[:len(essay_ele.essay) - 4] + ".comp", "w", encoding='utf-8')
        for token in essay_ele.tokens:
            file.write(token.get_token_annotation())
            if token.token == "." or token.token == "!" or token.token == "?":
                file.write("\n")
        file.close()


if __name__ == "__main__":
    load_essays = True
    load_annotations = False
    if load_essays:
        essay_list = parse_essays("")   # TODO PATH
            #r"...\data\Argument_Component_Identification_Stab\brat-project-final")
        df_essay = pd.DataFrame.from_records([essay.as_dict() for essay in essay_list])
        # print(df_essay.head())
    if load_annotations:
        annotations_Stab = parse_annotations_Stab("")   # TODO PATH
            # "...\data\Argument_Component_Identification_Stab\brat-project-final")
        # print(annotations_Stab.head())
        ''' only req. if no df if returned but annotations as list'''
        # analyze_annotations_Stab(annotations_Stab)
        # ACI_Annotation_Stab = pd.DataFrame([annotation.as_dict() for annotation in annotations_Stab])
        # print(ACI_Annotation_Stab.head())

    # go trough each essay
    annotations = parse_annotations_Stab_perFile("")   # TODO PATH
        #"...\data\Argument_Component_Identification_Stab\brat-project-final")
    join_argument_annotations_and_texts(essay_list, annotations)
    # write annotations
    # write_annotated_essays(essay_list)
