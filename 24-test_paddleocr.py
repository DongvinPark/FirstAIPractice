from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize PaddleOCR (minimal config for v3.0.1+)
ocr = PaddleOCR(
    lang='ch'
    )  # 'ch' for Chinese

# Load image
img_path = './data/ch1.jpg'
image = cv2.imread(img_path)

# Run OCR
result = ocr.predict(img_path)

myresult = result[0]

for res in result:
    #res.print()
    res.save_to_img("out")
    res.save_to_json("output")

if 1: 
    # Extract data from the result object
    res = result[0].json['res']
    texts = res['rec_texts']#.rec_texts    # List of recognized texts
    confidences = res['rec_scores']  # List of confidence scores
    boxes = res['rec_boxes']    # List of detection boxes [x1,y1,x2,y2]

    # Process each detected text
    for idx, (text, confidence, box) in enumerate(zip(texts, confidences, boxes)):
        print(f"Text {idx + 1}: {text} | Confidence: {confidence:.2f}| coord {box}")

        cv2.rectangle(image, (box[0],box[1]), (box[2], box[3]), (255,0,0), 2)


        # Convert box from [x1,y1,x2,y2] to 4 points [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        box_4pt = [
            box[0],
            box[1],
            box[2],
            box[3]
            #[box[0], box[1]],  # Top-left
            #[box[2], box[1]],  # Top-right
            #[box[2], box[3]],  # Bottom-right
            #[box[0], box[3]]   # Bottom-left
        ]

        # Draw bounding box (green)
        
        #cv2.polylines(
        #    image,
        #    [np.array(box_4pt, dtype=np.int32)],
        #    isClosed=True,
        #    color=(0, 255, 0),
        #    thickness=2
        #)

        # Add text label (red)
        if 0:
            cv2.putText(
                image,
                text,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # Save visualized result
    cv2.imwrite('ocr_result_latest.jpg', image)
    print("Results saved to 'ocr_result_latest.jpg'")