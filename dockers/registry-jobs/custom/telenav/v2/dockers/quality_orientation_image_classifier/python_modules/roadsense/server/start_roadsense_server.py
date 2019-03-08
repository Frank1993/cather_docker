from flask import Flask, current_app, request, jsonify
import pickle
import json
import os
import argparse


TRIP_ID_PARAM = "trip_id"
BASE_PATH = "/home/docker/apollo/roadsense/"

app = Flask(__name__)
    
@app.route('/upload', methods=['POST'])
def predict():
    
    try:
        trip_json = request.get_json()
        trip_id = trip_json[TRIP_ID_PARAM]

        output_path = os.path.join(BASE_PATH,'{}.json'.format(trip_id))

        with open(output_path, 'w') as outfile:
            json.dump(trip_json, outfile, indent=4, sort_keys=True)

        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    
    except Exception as e:
        print(e)
        
    return json.dumps({'success':False}), 500, {'ContentType':'application/json'}


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    return parser.parse_args()

if __name__ == '__main__':
    args = __parse_args()
    
    os.makedirs(BASE_PATH, exist_ok=True)
    
    app.run(host='0.0.0.0', port=args.port, debug=False)