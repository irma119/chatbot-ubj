import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from twilio.twiml.messaging_response import MessagingResponse
import json
from dotenv import load_dotenv
from chat import get_response
# Load environment variables from .env file
load_dotenv()
# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict():
    # Retrieve the incoming message from Twilio
    incoming_msg = request.values.get('Body', '').lower()
    # TODO: Add validation for incoming_msg if needed
    # Get a response using your chat function
    response = get_response(incoming_msg)
    # Create a Twilio response
    twilio_resp = MessagingResponse()
    twilio_resp.message(response)

    return str(twilio_resp)

if __name__ == "__main__":
    app.run(debug=True)