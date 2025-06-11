import requests

# API_KEY = 'my key' # 주석을 해제하고 사용한다.
url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

params = {
    'q': '안녕하세요',
    'source': 'ko',
    'target': 'en',
    'format': 'text'
}

response = requests.post(url, data=params)
result = response.json()

print("Translated text:", result['data']['translations'][0]['translatedText'])