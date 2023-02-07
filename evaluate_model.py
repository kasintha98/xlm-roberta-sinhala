from simpletransformers.ner import NERModel
from get_configs import Configs

label = Configs.get_labels()
args = Configs.get_args()
model = NERModel('xlmroberta', 'xlm-roberta-base', labels=label, args=args, use_cuda=False)
test_data = Configs.get_dataset()

result, model_outputs, preds_list = model.eval_model(test_data)

print(result)