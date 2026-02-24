import os
import json
import urllib.request
import urllib.error

API_KEY = os.environ.get("OPENAI_API_KEY") 
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent"

def test_tokens(num_tokens):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # ~4 chars per token
    dummy_text = "word " * int(num_tokens * 0.8)

    data = {
        "model": "rlm-internal.rlm-core-v1",
        "messages": [
            {"role": "user", "content": f"Repeat this: {dummy_text}"}
        ],
        "max_tokens": 100
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        response = urllib.request.urlopen(req)
        print(f"Success at ~{num_tokens} tokens")
    except urllib.error.HTTPError as e:
        error_info = e.read().decode('utf-8')
        print(f"Failed at ~{num_tokens} tokens with {e.code}: {error_info}")

test_tokens(100000)
test_tokens(120000)
test_tokens(130000)
