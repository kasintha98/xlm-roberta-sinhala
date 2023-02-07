import pandas as pd
import os as os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel
from get_configs import Configs

data = pd.read_csv("ner_dataset1.4.csv", encoding="utf8")

data["Sentence #"] = LabelEncoder().fit_transform(data["Sentence #"])
data.rename(columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"}, inplace=True)
data["labels"] = data["labels"].str.upper()

X = data[["sentence_id", "words"]]
Y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.01)

# building up train data and test data
train_data = pd.DataFrame({"sentence_id": x_train["sentence_id"], "words": x_train["words"], "labels": y_train})
test_data = pd.DataFrame({"sentence_id": x_test["sentence_id"], "words": x_test["words"], "labels": y_test})

# label = data["labels"].unique().tolist()
label = Configs.get_labels()

args = Configs.get_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model = NERModel('xlmroberta', 'xlm-roberta-large',labels=label,args =args, use_cuda= False)

# model = NERModel('bert', 'bert-base-cased', labels=label, args=args, use_cuda=False)

model = NERModel('xlmroberta', 'xlm-roberta-base', labels=label, args=args, use_cuda=False)

model.train_model(train_data, eval_data=test_data, acc=accuracy_score)
