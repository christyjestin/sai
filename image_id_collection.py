import json
import pickle
import sys

# iterate through caption data to form a list of image ids and then
# save the list via pickling; this is needed for training since the image ids
# aren't a contiguous set of integers
def main(train_file, test_file):
    with open(train_file, mode = 'r') as f:
        train_data = json.load(f)
    with open(test_file, mode = 'r') as f:
        test_data = json.load(f)

    train_ids = [img["id"] for img in train_data["images"]]
    test_ids = [img["id"] for img in test_data["images"]]

    with open("train_ids.pkl", mode = "wb") as f:
        pickle.dump(train_ids, f, pickle.HIGHEST_PROTOCOL)
    with open("test_ids.pkl", mode = "wb") as f:
        pickle.dump(test_ids, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    help = """Please provide the caption train json file and the caption test json file in that order"""
    if len(sys.argv) != 3:
        print(help)
    else:
        main(*sys.argv[1:])