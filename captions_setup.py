import numpy as np
import json
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors as KV
import re, string
import sys

glove = KV.load_word2vec_format('glove.840B.300d.txt', binary = False)

punc_regex = '([' + string.punctuation + '])'
def add_space_around_punc(s):
    return re.sub(punc_regex, r' \1 ', s)

# take in caption data json filename and return cleaned dictionary
# with image ids as the key and a list of cleaned captions as the value
# N.B. cleaned captions have spaces around all punctation, so they'll separate when we call caption.split()
def process_caption_data(json_file):
    with open(json_file, mode = 'r') as f:
        captions_data = json.load(f)
    captions_dict = defaultdict(list)
    for annot in captions_data['annotations']:
        captions_dict[annot['image_id']].append(add_space_around_punc(annot['caption']))
    return captions_dict

# process captions to calculate inverse frequencies
def calc_inv_frequencies(captions_dict):
    frequencies = defaultdict(lambda: 0)
    total_count = 0
    for id in captions_dict:
        for caption in captions_dict[id]:
            for word in caption.split():
                frequencies[word] += 1
                total_count += 1
    inv_frequencies = dict()
    for word in frequencies:
        inv_frequencies[word] = total_count / frequencies[word]
    return inv_frequencies

# represent captions as arrays using glove embeddings and inverse frequencies
# and then save these arrays to the given directory
def write_captions_array(captions_dict, inv_frequencies, array_dir):
    for id in captions_dict:
        for i, caption in enumerate(captions_dict[id]):
            seq = []
            # represent each word as an array with shape (301,) formed by 
            # appending the inverse frequency (scalar) to the glove embedding (array with shape (300,))
            for word in caption.split():
                if word in glove and word in inv_frequencies: seq.append(np.append(glove[word], inv_frequencies[word]))
            with open(f"{array_dir}/{id}_{i}.npy", mode = "wb") as f:
                np.save(f, np.array(seq))

def main(train_file, test_file, train_array_dir, test_array_dir):
    train_dict = process_caption_data(train_file)
    test_dict = process_caption_data(test_file)

    # only training data is used to generate inverse frequencies
    inv_frequencies = calc_inv_frequencies(train_dict)

    write_captions_array(train_dict, inv_frequencies, train_array_dir)
    write_captions_array(test_dict, inv_frequencies, test_array_dir)

if __name__ == "__main__":
    help = """Please provide the caption train json file, the caption test json file, the training array
        directory, and the testing array directory in that order"""
    if len(sys.argv) != 5:
        print(help)
    else:
        main(*sys.argv[1:])