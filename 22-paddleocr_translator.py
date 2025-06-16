import cv2
from paddleocr import PaddleOCR
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import requests

"""
이 코드를 우분투 22.04 에서 실행시키기 위해서는
>>> 파이썬 3.10 버전으로 콘다 env를 만들고 활성화 시킨 다음,
paddlepaddle 공식 홈페이지(링크 : https://www.paddlepaddle.org.cn/en/install/quick?docurl=undefined)에 들어가서 
Paddle Build : 2.6
your OS : Linux
Packdage : pip
Venders : CPU
Compute Platform : x86
옵션을 선택한 후 제공되는 명령어를 이용해서 paddlepaddle을 설치해야 한다.
>>> 그후, paddleocr을 pip 명령어를 써서 2.6.0.1 버전으로 설치해야 한다.
최종적으로 pip list 명령어를 실행해보면,
...
paddleocr                2.6.1.0
paddlepaddle             2.6.1
...
라는 버전 정보가 뜰 것이다.
>>> numpy 버전과 관련된 에러가 뜬다면, 해당 에러로그를 ChatGPT에게 전달해서 어느 코드를 고쳐야 되는지를 물어보면 된다.

paddleocr, paddlepaddle 버전이 3.0 이면 안 된다.
"""

# =======================
# STEP 1: OCR - Detect Chinese Text
# =======================
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Chinese OCR
img_path = "./data/ch1.jpg"  # <- 여기에 이미지 경로 지정
ocr_result = ocr.ocr(img_path, cls=True)

# =======================
# STEP 2: Install zh → ko Translation Model (only once)
# =======================
# dongvin, there is no direct way from chinese to korean. So break it down.
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


# =======================
# STEP 4: Draw Boxes + Translation
# =======================
from_to_code = [("zh","en"), ("en","ko")]

image_pil = Image.open(img_path)
draw = ImageDraw.Draw(image_pil)

# dongvin, make sure to run "sudo apt install fonts-nanum" in linux
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path,size=15)

chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
for line in ocr_result:
    for box, (text, confidence) in line:
        if confidence < 0.4:  # low confidence skip
            print(f"dongvin, {text} has low confidence {confidence}, skip it")
            continue
        if chinese_char_pattern.search(text) is None: 
            #print(f"dongvin, {text} is not a chinese text. skip it")    
            continue
        # meta
        #translator of NLLB (Meta)

        from_code = 'zh'
        to_code = 'ko'
        translated = do_translate(text, from_code, to_code)

        print(f"dongvin, from: {from_code}:{text}, to {to_code}:{translated}, confience: {confidence}")

        # Draw bounding box
        _box =(box[0], box[2])
        draw.rectangle(_box, outline="blue", width=2)

        x, y = box[-1]
        draw.text((x, y), translated, font=font, fill=(255, 0, 0))

# =======================
# STEP 5: Save or Show
# =======================
#image_pil.show()
image_pil.save('./out/ch-kor-translated.jpg')
