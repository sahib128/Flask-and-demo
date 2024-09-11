import requests

# Define the URL of your Flask server
flask_server_url = 'http://172.16.22.230:5000/query'

# Define the query and context
query_text = "What is the topic of the context?"
context_text = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data without being explicitly programmed."

# Prepare the payload for the request
payload = {
    'query': query_text,
    'context': context_text,
}

# Send the POST request to the Flask server
response = requests.post(flask_server_url, json=payload)

# Check the response from the server
if response.status_code == 200:
    response_data = response.json()
    print("Response from Flask server:")
    print(response_data['response'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
