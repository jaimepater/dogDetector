import super_gradients
import cv2
from super_gradients.training import models
import supervision as sv
from debounce import debounce
import requests
import os
import uuid
from dotenv import load_dotenv


load_dotenv()
API_BASE_URL = os.environ.get('API_BASE_URL')
SECRET = os.environ.get('SECRET')

@debounce(5)
def send_frame_as_image(img, is_big_dog):
    print("entrooo")
    # Convert the frame to bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    url = f'{API_BASE_URL}/dog'
    headers = {
        'Authorization': f'Bearer {SECRET}'
    }
    if is_big_dog:
        payload_type = 'big-dog'
    else:
        payload_type = 'dog'

    # Request payload
    payload = {
        'type': payload_type
    }
    image_name = str(uuid.uuid4())
    print(url, headers, payload)
    files = {
        'image': (image_name, img_encoded.tobytes(), 'image/jpeg')
    }
    response = requests.post(url, headers=headers, data=payload, files=files)
    if response.status_code == 200:
        # Request was successful
        print('API request successful.')
        print(response.json())


def get_camera():
    x = 1920
    y = 1080
    fps = 30
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, x)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, y)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(width, height, fps)
    return capture


def detector_bg(img, model, model_bd):
    pred = list(model.predict(img))
    have_big_dog = False
    detections = sv.Detections.from_yolo_nas(pred[0])
    detections = detections[detections.confidence > 0.5]
    detections = detections[detections.class_id == 16]

    if len(detections) > 0:
        bbboxes = detections.xyxy
        for i, b in enumerate(bbboxes):
            bbox = bbboxes[i].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        pred_bg = list(model_bd.predict(img))
        detections_bg = sv.Detections.from_yolo_nas(pred_bg[0])
        detections_bg = detections_bg[detections_bg.confidence > 0.5]
        detections_bg = detections_bg[detections_bg.class_id == 0]
        if len(detections_bg) > 0:
            have_big_dog = True
            bbboxes_bg = detections_bg.xyxy
            for j, d in enumerate(bbboxes_bg):
                bbox_bg = bbboxes_bg[j].astype(int)
                cv2.rectangle(frame, (bbox_bg[0], bbox_bg[1]), (bbox_bg[2], bbox_bg[3]), (0, 255, 0), 2)
        send_frame_as_image(frame.copy(), have_big_dog)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    capture = get_camera()
    yolo_nas = super_gradients.training.models.get("yolo_nas_m", pretrained_weights="coco")
    model_bd = models.get("yolo_nas_l",
                          num_classes=2,
                          checkpoint_path="bg_nas_l.pth")
    while True:
        ret, frame = capture.read()
        if ret:
            detector_bg(frame, yolo_nas, model_bd)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
