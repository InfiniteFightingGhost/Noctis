from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/api/epochs:ingest-device', methods=['POST'])
def ingest_device_epochs():
    print("------")
    print("Received a request...")

    # Print authorization header
    auth_header = request.headers.get('Authorization')
    print(f"Authorization Header: {auth_header}")

    # Get and print the JSON payload
    try:
        data = request.get_json()
        print("Received JSON data:")
        print(json.dumps(data, indent=2))
        
        # Check for device ID
        if 'device_external_id' in data and 'epochs' in data:
            print("Validation: Looks like a valid payload!")
            return jsonify({"message": "Success!"}), 200
        else:
            print("Validation: Payload missing required fields.")
            return jsonify({"error": "Invalid payload"}), 400

    except Exception as e:
        print(f"Error processing JSON: {e}")
        return jsonify({"error": "Could not parse JSON"}), 400

if __name__ == '__main__':
    print("Starting mock Noctis API server...")
    print("Listening for POST requests at http://0.0.0.0:5000/api/epochs:ingest-device")
    # To find your IP, use 'ipconfig' (Windows) or 'ifconfig'/'ip addr' (Mac/Linux)
    app.run(host='0.0.0.0', port=5000)
