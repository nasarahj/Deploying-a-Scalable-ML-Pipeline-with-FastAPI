import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# TODO: print the status code
print("Get status code:",r.status_code)
# TODO: print the welcome message
print("Get Response:", r.json()["message"])



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data)

# TODO: print the status code
print("POST Status Code:", r.status_code)
try:
    print("POST Result:", r.json())
except requests.exceptions.JSONDecodeError:
    print("POST Raw Text:", r.text)   # shows FastAPI error details
# TODO: print the result
print("POST Result:", r.json()["result"])
