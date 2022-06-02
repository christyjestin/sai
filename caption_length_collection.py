import pickle
import json
from collections import defaultdict
import re, string
import sys

# N.B. I copied this section of code over from captions_setup.py since that file loads glove embeddings
# that just takes too long, and glove embeddings are not necessary for this task
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

# process captions to calculate the length of each cpation
def calc_caption_lengths(ids, captions_dict):
    caption_lengths = []
    for id in ids:
        for caption in captions_dict[id][:5]:
            caption_lengths.append(len(caption.split()))
    return caption_lengths

# process json files and write caption lengths
def main():
    train_dict = process_caption_data("captions_train.json")
    test_dict = process_caption_data("captions_test.json")

    # retrieve saved id lists
    with open("train_ids.pkl", mode = "rb") as f:
        train_ids = pickle.load(f)
    with open("test_ids.pkl", mode = "rb") as f:
        test_ids = pickle.load(f)

    with open("train_caption_lengths.pkl", mode = "wb") as f:
        pickle.dump(calc_caption_lengths(train_ids, train_dict), f, pickle.HIGHEST_PROTOCOL)
    with open("test_caption_lengths.pkl", mode = "wb") as f:
        pickle.dump(calc_caption_lengths(test_ids, test_dict), f, pickle.HIGHEST_PROTOCOL)

# iterate through caption data to form a list of caption lengths and save the list via pickling
# this is needed to batch captions of similar lengths togeter
if __name__ == "__main__":
    help = "This script requires no arguments"
    if len(sys.argv) != 1:
        print(help)
    else:
        main()