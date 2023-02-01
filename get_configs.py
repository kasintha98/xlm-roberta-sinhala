from simpletransformers.ner import NERArgs


class Configs:

    @staticmethod
    def get_args():
        args = NERArgs()
        args.num_train_epochs = 1
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
            "TIME"
        ]
        return labels
