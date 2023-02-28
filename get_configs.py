from simpletransformers.ner import NERArgs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Configs:

    @staticmethod
    def get_args():
        args = NERArgs()
        args.num_train_epochs = 5
        args.learning_rate = 1e-4
        args.overwrite_output_dir = True
        args.train_batch_size = 32
        args.eval_batch_size = 32
        args.use_multiprocessing = False
        args.use_multiprocessing_for_evaluation = False
        return args

    @staticmethod
    def get_labels():
        labels = [
            "LOCATION",
            "PERSON",
            "ORGANIZATION",
            "DATE",
            "TIME",
            "OTHER"
        ]
        return labels

    @staticmethod
    def get_dataset():
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

        return train_data
