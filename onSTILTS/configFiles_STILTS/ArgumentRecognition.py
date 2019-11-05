import pandas as pd
import xml.etree.ElementTree as et

def parse_XML(xml_file, df_cols):
    """Parse the input XML file and store the result in a pandas
    DataFrame with the given columns.

    The first element of df_cols is supposed to be the identifier
    variable, which is an attribute of each node element in the
    XML data; other features will be parsed from the text content
    of each sub-element.
    """

    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []

    for node in xroot:
        args = []
        args.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]:
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else:
                res.append(None)
        rows.append({df_cols[i]: res[i]
                     for i, _ in enumerate(df_cols)})

    out_df = pd.DataFrame(rows, columns=df_cols)

    return out_df

path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Argument Recognition\comarg.v1\comarg"


arg_recognition_corpus = parse_XML(path +"\GM.xml", ["id", "email", "grade", "age"])


