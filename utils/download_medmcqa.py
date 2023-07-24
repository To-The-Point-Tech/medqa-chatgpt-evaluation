import json
from os import mkdir
from os.path import join, exists
import datasets

if __name__ == "__main__":
    dataset = datasets.load_dataset("medmcqa")
    for split, file_name in zip(["validation", "train", "test"], ["dev.json", "train.json", "test.json"]):
        subset = dataset[split]
        dataset_path = join("data", "medmcqa")
        if not exists(dataset_path):
            mkdir(dataset_path)
        with open(join(dataset_path, file_name), "w") as f:
            for line in subset:
                f.write(json.dumps({
                    "id": line["id"], 
                    "sent1":  line["question"], 
                    "sent2": "",
                    "ending0": line["opa"], 
                    "ending1": line["opb"], 
                    "ending2": line["opc"], 
                    "ending3": line["opd"],
                    "label": line["cop"]
                }) + "\n")
