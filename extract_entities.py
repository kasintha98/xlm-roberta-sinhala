from simpletransformers.ner import NERModel
from get_configs import Configs
from fuzzysearch import find_near_matches

label = Configs.get_labels()
args = Configs.get_args()
# model = NERModel('bert', 'bert-base-cased', labels=label, args=args, use_cuda=False)
model = NERModel('xlmroberta', 'xlm-roberta-base', labels=label, args=args, use_cuda=False)


class ExtractEntities:

    @staticmethod
    def tag_entities_from_model(text):
        prediction, model_output = model.predict([text])
        # prediction = [[{'ආසියානු': 'PERSON'}, {'ක්\u200dරිකට්': 'PERSON'}, {'ශුර': 'PERSON'}, {'ශ්\u200dරී':
        # 'PERSON'}, {'ලංකා': 'PERSON'}, {'කණ්ඩායම': 'PERSON'}, {'සහ': 'PERSON'}, {'සත්කාරක': 'PERSON'}, {'ඉන්දීය':
        # 'PERSON'}, {'කණ්ඩායම': 'DATE'}]]
        # entities = [{"text": [*ent][0], "entity": list(ent.values())[0], "value": [*ent][0], "extractedBy": "xlm-roberta_model"} for ent in prediction[0]]
        entities = []

        for ent in prediction[0]:
            start_and_end = ExtractEntities.get_start_and_end_char(text, [*ent][0])
            if start_and_end:
                entity = {"text": [*ent][0], "entity": list(ent.values())[0], "value": [*ent][0], "startChar": start_and_end[0].start,
                                 "endChar": start_and_end[0].end, "extractedBy": "xlm-roberta_model"}
                entities.append(entity)

        return entities

    @staticmethod
    def get_start_and_end_char(full_text, matched_text):
        near_match = find_near_matches(matched_text, full_text, max_l_dist=1)
        return near_match
