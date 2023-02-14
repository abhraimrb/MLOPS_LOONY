import requests
import json

test_sample = json.dumps({'data': [
    [56, 1, 39.82, 0, 0, 2, 11090.7178], 
    [23, 0, 34.4, 0, 0, 3, 1826.843]
]})
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0
