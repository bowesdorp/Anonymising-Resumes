import pandas as pd
import spacy
from spacy.training import offsets_to_biluo_tags
import os
nlp = spacy.load("en_core_web_sm")
prefixes = ['\\n', '\\t'] + nlp.Defaults.prefixes


prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search

def convert_to_dataturks(dataframe):
    length = dataframe.shape[0]
    for i in range(0, length):
        entities = []
        annotations = dataframe['annotation'][i]
        for annotation in annotations:
            obj = {
                "label": [annotation['label']],
                "points": [
                    {
                        "start": annotation['start'],
                        "end": annotation['end'],
                        "text": annotation['text']
                    }
                ]
            }
            entities.append(obj)
        dataframe['annotation'][i] = entities

    return dataframe


def get_entities(df):
    entities = []

    for i in range(len(df)):
        entity = []

        for annot in df['annotation'][i]:
            try:
                ent = annot['label'][0]
                start = annot['points'][0]['start']
                end = annot['points'][0]['end'] + 1
                entity.append((start, end, ent))
            except:
                pass

        entity = mergeIntervals(entity)
        entities.append(entity)
    return entities


def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)

    return merged


def get_train_data(df):
    tags = []
    sentences = []

    for i in range(len(df)):
        text = df['document'][i]
        entities = df['entities'][i]

        doc = nlp(text)

        tag = offsets_to_biluo_tags(doc, entities)
        tmp = pd.DataFrame([list(doc), tag]).T
        loc = []
        for i in range(len(tmp)):
            if tmp[0][i].text == '.' and tmp[1][i] == 'O':
                loc.append(i)
        loc.append(len(doc))

        last = 0
        data = []
        for pos in loc:
            data.append([list(doc)[last:pos], tag[last:pos]])
            last = pos

        for d in data:
            tag = ['O' if t == '-' else t for t in d[1]]
            if len(set(tag)) > 1:
                sentences.append(d[0])
                tags.append(tag)

    return sentences, tags

def iob_to_csv():
    df = pd.DataFrame(columns=['Sentence #', 'Word', 'POS', 'Tag'])

    print(df)
    counter = 1
    path = './data/tagged/1-599/'
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            counter2 = 1
            split = filename.split('.')
            if split[1] == 'txt':
                f = open(path + filename, "r")
                lines = f.readlines()
                for line in lines:
                    txt_split = line.split(',')
                    if counter2 > 1:
                        if len(txt_split) == 3:
                            df = df.append({
                                'Sentence #': 'Sentence ' + str(counter),
                                'Word': txt_split[0],
                                'POS': txt_split[1],
                                'Tag': txt_split[2].replace('\n', '')
                            }, ignore_index=True)
                    counter2 += 1
                f.close()
            print(df.shape)
            counter += 1
    df.to_csv(r'./data_1-100.csv', index=False, header=True)