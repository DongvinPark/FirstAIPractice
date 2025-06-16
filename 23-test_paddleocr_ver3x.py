# This code is based on the latest code of paddleOCR (3.0.1)
# refer to https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file
# pip install paddleocr
# pip install paddlepaddle

"""
이 코드를 우분투 22.04 에서 실행시키기 위해서는
>>> 파이썬 3.10 버전으로 콘다 env를 만들고 활성화 시킨 다음,
paddlepaddle 공식 홈페이지(링크 : https://www.paddlepaddle.org.cn/en/install/quick?docurl=undefined)에 들어가서 
Paddle Build : 3.0
your OS : Linux
Packdage : pip
Venders : CPU
Compute Platform : x86
옵션을 선택한 후 제공되는 명령어를 이용해서 paddlepaddle을 설치해야 한다.
>>> 그후, pip install paddleocr명령어를 실행하고 나머지 필요한 디펜던시들을
pip install ~~ 명령어로 설치해줘야 한다.

pip list cmd's result will show this lines.
paddleocr                3.0.1
paddlepaddle             3.0.0
"""

from paddleocr import PaddleOCR
import json
from utils import get_filename_only, is_chinese
import os
from PIL import Image, ImageDraw, ImageFont

import requests
import matplotlib.pyplot as plt

input_image = './data/ch1.jpg' # it's okay to use an image on a website.
name = get_filename_only(input_image)
out_path = './out'
out_name = out_path+'/'+name+'_out'

#------------------
# Chinese Text Detection and location.
#-----------------
if not os.path.exists(out_name+'.json'):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ch')

    result = ocr.predict(input=input_image)
    # dongvin, this is the basic rule as written in the github
    # so it's better to handle the resultant json to isolate the file handling
    # from the detection and generation of data.
    for res in result:
        res.save_to_img(out_name+'.jpg')
        res.save_to_json(out_name+'.json')

#------------------
# Json File for Chinese Text, Location, Score(=confidence)
#-----------------
selected_keys=[]
with open(out_name+'.json', 'r') as file:
    data = json.load(file)
    # just guess the item's key. this is subject to changing.
    _keys=['texts','boxes','scores'] 
    for _key in _keys:
        matches = [key for key in data.keys() if _key in key]
        if matches: selected_keys.extend(matches)
    print(*selected_keys)        

texts, boxes, scores = [data.get(key) for key in selected_keys]

#------------------
# Translate via Google Cloud Translate API
#-----------------
with open('/home/alphaai/Documents/SECRET/GoogleTranslationAPIKey.txt', 'r') as file:
    API_KEY = file.readline().strip()
translation_url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

def do_translate(input_txt, input_country_code, target_country_code):
    params = {
        'q': input_txt,
        'source': input_country_code,
        'target': target_country_code,
        'format': 'text'
    }

    try:
        response = requests.post(translation_url, data=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['data']['translations'][0]['translatedText']

    except requests.exceptions.HTTPError as e:
        print(f"[HTTP Error] {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[Request Error] {e}")
    except (KeyError, IndexError) as e:
        print(f"[Response Parsing Error] {e}")
    except Exception as e:
        print(f"[Unexpected Error] {e}")

    return ""  # Return empty string on any failure

#------------------
# Drawing
#-----------------
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path,size=15)

image_pil = Image.open(input_image)
draw = ImageDraw.Draw(image_pil)

for text,box,score in zip(texts,boxes,scores):
    if not is_chinese(text) or score<0.4: continue

    # boxing
    draw.rectangle(box, outline='blue',width=2)

    # translation via Googla Cloud Translation API
    translated = do_translate(text, 'zh', 'ko')
    print(f"dongvin, {text}:{translated}")

    x,y = box[0], box[3]
    draw.text((x, y), translated, font=font, fill=(255, 0, 0))
   

plt.imshow(image_pil)
plt.axis('off')
plt.show()