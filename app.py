from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from extract_entities import ExtractEntities

# create an instance of flask
app = Flask(__name__)
# creating an API object
api = Api(app)


# For Post request to http://localhost:6000/check-entities
# Command Start Server On Port:6000 -> flask run -h localhost -p 6000
class CheckEntities(Resource):
    def post(self):
        if request.is_json:
            text = request.json['text']
            extracted_model_entities = ExtractEntities.tag_entities_from_model(text)
            # return a json response
            return make_response(jsonify({'text': text, 'extractedEntitiesFromModel': extracted_model_entities}), 200)
        else:
            return {'error': 'Request must be JSON'}, 400


api.add_resource(CheckEntities, '/check-entities')

#
if __name__ == '__main__':
    app.run(host="localhost", port=6000, debug=True)
