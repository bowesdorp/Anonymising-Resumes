import pandas as pd
import nltk
import json
import ast
import numpy as np
import torch
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_label(offset, labels):
    if offset[0] == 0 and offset[1] == 0:
        return 'O'
    for label in labels:
        if offset[1] >= label[0] and offset[0] <= label[1]:
            return label[2]
    return 'O'


def process_resume(data, tokenizer, tag2idx, max_len, is_test=False):
    tok = tokenizer.encode_plus(data[0], max_length=max_len, return_offsets_mapping=True)
    curr_sent = {'orig_labels': [], 'labels': []}

    padding_length = max_len - len(tok['input_ids'])

    if not is_test:
        labels = data[1]['entities']
        labels.reverse()
        for off in tok['offset_mapping']:
            label = get_label(off, labels)
            curr_sent['orig_labels'].append(label)
            curr_sent['labels'].append(tag2idx[label])
        curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
    return curr_sent

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


class ResumeDataset(Dataset):
    def __init__(self, resume, tokenizer, tag2idx, max_len, is_test=False):
        self.resume = resume
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, idx):
        data = process_resume(self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'orig_label': data['orig_labels']
        }


if __name__ == '__main__':
    df = pd.read_json(r'data/tagged/6e332969-b5d7-400c-b05d-d348465e8117.json', orient='records')
    for i in range(len(df)):
        df["document"][i] = df["document"][i].replace("\n", " ").replace("\t", " ")
    length = df.shape[0]
    spacy_format = []

    for i in range(0, length):
        entities = []
        annotations = df['annotation'][i]
        for annotation in annotations:
            # entity = (annotation['start'], annotation['end'], annotation['label'])
            #         obj = {
            #             "entities": [annotation['label']],
            #             "points": [
            #                 {
            #                     "start": annotation['start'],
            #                     "end": annotation['end'],
            #                     "text": annotation['text'].replace("\n", " ").replace("\t", " ")
            #                 }
            #             ]
            #         }
            entities.append(entity)
        labels = {"entities": entities}
        document = df["document"][i]
        spacy_format.append((document, labels))
    data = trim_entity_spans(spacy_format)





    cleanedDF = pd.DataFrame(columns=["Sentences"])
    sum1 = 0
    for i in range(len(data)):
        start = 0
        emptyList = ["EMPTY"] * len(data[i][0].split())
        numberOfWords = 0
        lenOfString = len(data[i][0])
        strData = data[i][0]
        strDictData = data[i][1]
        lastIndexOfSpace = strData.rfind(' ')
        for i in range(lenOfString):
            if (strData[i] == " " and strData[i + 1] != " "):
                for k, v in strDictData.items():
                    for j in range(len(v)):
                        entList = v[len(v) - j - 1]
                        if (start >= int(entList[0]) and i <= int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            break
                        else:
                            continue
                start = i + 1
                numberOfWords += 1
            if (i == lastIndexOfSpace):
                for j in range(len(v)):
                    entList = v[len(v) - j - 1]
                    if (lastIndexOfSpace >= int(entList[0]) and lenOfString <= int(entList[1])):
                        emptyList[numberOfWords] = entList[2]
                        numberOfWords += 1
        cleanedDF = cleanedDF.append(pd.Series([emptyList], index=cleanedDF.columns), ignore_index=True)
        sum1 = sum1 + numberOfWords


    totalNumWords = [len(one_comment.split()) for one_comment in df["document"]]
    # plt.hist(totalNumWords)
    # plt.show()


    MAX_LEN = 300
    bs = 16

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(df["document"])

    tokenized_texts = tokenizer.texts_to_sequences(df["document"])

    input_ids = pad_sequences(tokenized_texts,
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    all_labels = []
    for entry in cleanedDF["Sentences"]:
        all_labels += list(set(entry))
    tags_vals = list(set(all_labels)) + ['UNKOWN']

    tag2idx = {t: i for i, t in enumerate(tags_vals)}
    idx2tag = {i: t for i, t in enumerate(tags_vals)}

    train_data, val_data = data[:16], data[16:]

    print(train_data)

    train_d = ResumeDataset(train_data, tokenizer, tag2idx, MAX_LEN)
    val_d = ResumeDataset(val_data, tokenizer, tag2idx, MAX_LEN)
    #
    # tag2idx = {t: i for i, t in enumerate(all_labels)}
    #
    # labels = cleanedDF["Sentences"].tolist()
    # tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
    #                      maxlen=MAX_LEN, value=tag2idx["EMPTY"], padding="post",
    #                      dtype="long", truncating="post")
    #
    # attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    #
    # tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
    #                                                             random_state=2018, test_size=0.2)
    # tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
    #                                              random_state=2018, test_size=0.2)
    #
    # tr_inputs = torch.tensor(tr_inputs)
    # val_inputs = torch.tensor(val_inputs)
    # tr_tags = torch.tensor(tr_tags)
    # val_tags = torch.tensor(val_tags)
    # tr_masks = torch.tensor(tr_masks)
    # val_masks = torch.tensor(val_masks)
    #
    # train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    #
    # valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    # valid_sampler = SequentialSampler(valid_data)
    # valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    #
    # model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
    #
    # FULL_FINETUNING = True
    # if FULL_FINETUNING:
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ['bias', 'gamma', 'beta']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.0}
    #     ]
    # else:
    #     param_optimizer = list(model.classifier.named_parameters())
    #     optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    # optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    #
    # epochs = 5
    # max_grad_norm = 1.0
    #
    # for _ in range(epochs):
    #     # TRAIN loop
    #     model.train()
    #     tr_loss = 0
    #     nb_tr_examples, nb_tr_steps = 0, 0
    #     for step, batch in enumerate(train_dataloader):
    #         b_input_ids, b_input_mask, b_labels = batch
    #         # forward pass
    #         loss = model(b_input_ids, token_type_ids=None,
    #                      attention_mask=b_input_mask, labels=b_labels)
    #         print('loss', loss)
    #         # backward pass
    #         loss.backward()
    #         # track train loss
    #         tr_loss += loss.item()
    #         nb_tr_examples += b_input_ids.size(0)
    #         nb_tr_steps += 1
    #         # gradient clipping
    #         torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
    #         # update parameters
    #         optimizer.step()
    #         model.zero_grad()
    #     # print train loss per epoch
    #     print("Train loss: {}".format(tr_loss / nb_tr_steps))
    #     # VALIDATION on validation set
    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     predictions, true_labels = [], []
    #     for batch in valid_dataloader:
    #
    #         b_input_ids, b_input_mask, b_labels = batch
    #
    #         with torch.no_grad():
    #             tmp_eval_loss = model(b_input_ids, token_type_ids=None,
    #                                   attention_mask=b_input_mask, labels=b_labels)
    #             logits = model(b_input_ids, token_type_ids=None,
    #                            attention_mask=b_input_mask)
    #         logits = logits.detach().cpu().numpy()
    #         label_ids = b_labels.to('cpu').numpy()
    #         predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    #         true_labels.append(label_ids)
    #
    #         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    #
    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy
    #
    #         nb_eval_examples += b_input_ids.size(0)
    #         nb_eval_steps += 1
    #     eval_loss = eval_loss / nb_eval_steps
    #     print("Validation loss: {}".format(eval_loss))
    #     print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    #
    #     # print(predictions)
    #     # print(true_labels)
    #     # true_labels = true_labels[0]
    #     # print('-----------')
    #     # for j in range(len(true_labels)):
    #     #     for k in range(len(true_labels[j])):
    #     #         print(true_labels[j][k])
    #     #         print(predictions[j][k])
    #     #         predictions[j][k] = all_labels[predictions[j][k]]
    #     #         true_labels[j][k] = all_labels[true_labels[j][k]]
    #     pred_tags = [[all_labels[p_i] for p in predictions for p_i in p]]
    #     valid_tags = [[all_labels[l_ii] for l in true_labels for l_i in l for l_ii in l_i]]
    #
    #     # pred_tags = [int(p_i) for p in predictions for p_i in p]
    #     # valid_tags = [int(l_ii) for l in true_labels for l_i in l for l_ii in l_i]
    #
    #     print('preds', predictions)
    #     print('-----------')
    #     print('trags', true_labels)
    #     f1_s = f1_score(pred_tags, valid_tags)
    #     print("F1-Score: {}".format(f1_s))
    #
