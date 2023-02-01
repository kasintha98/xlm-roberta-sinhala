from simpletransformers.ner import NERModel
from get_configs import Configs

label = Configs.get_labels()
args = Configs.get_args()
model = NERModel('bert', 'bert-base-cased', labels=label, args=args, use_cuda=False)


class ExtractEntities:

    @staticmethod
    def tag_entities_from_model(text):
        prediction, model_output = model.predict([text])
        # prediction = [[{'ආසියානු': 'PERSON'}, {'ක්\u200dරිකට්': 'PERSON'}, {'ශුර': 'PERSON'}, {'ශ්\u200dරී':
        # 'PERSON'}, {'ලංකා': 'PERSON'}, {'කණ්ඩායම': 'PERSON'}, {'සහ': 'PERSON'}, {'සත්කාරක': 'PERSON'}, {'ඉන්දීය':
        # 'PERSON'}, {'කණ්ඩායම': 'DATE'}]]
        entities = [{"text": [*ent][0], "entity": list(ent.values())[0], "value": [*ent][0]} for ent in prediction[0]]
        return entities
