# Importing all the necessary libraries
import math
import cv2
import requests
from supervision.draw.color import Color
from ultralytics import YOLO
import supervision as sv
import argparse
# import bytetrack 
import base64





def send_sms(cam, image_file, className, conf):

    """
    send_sms -> This method is responsible for alerting the municipality about the detected hazard via 
    an SMS message.
    The message includes the camera location, the detected hazard image, the hazard class/type, 
    and the confidence score of the detection process.

    """
    # 1) Since the 'Gateway' SMS API does not support image files, we first need to convert 
    # the image file to a URL using imgBB API.

    # Our API key from imgBB
    api_key = '023a2a1fd5cadc3a4dc7f73b891268d6'

    # Read the image file and encode it in 'base64'
    # Open the image file in binary mode, encode it in base64, and decode it as UTF-8
    # This prepares the image data for safe transmission as text-based data,
    # ensuring compatibility with various systems and protocols.
    
    with open(image_file, 'rb') as file:
        image_data = base64.b64encode(file.read()).decode('utf-8')

    # Define the API endpoint URL
    url = "https://api.imgbb.com/1/upload"

    # Define the payload data
    payload = {
        "key": api_key,
        "image": image_data
    }

    # Make a POST request to upload the image on imgBB
    response = requests.post(url, data=payload)

    # Parse the JSON response
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200 and data['success']:

        # Get the URL of the hosted image
        image_url = data['data']['url']
        print(f"Image uploaded successfully. URL: {image_url}")

    else:
        print("Failed to upload the image.")
    

    # 2) Now, we need to construct the SMS message and send it to the concerned party.

    # The URL for SMS gateway
    sms_gateway_url = "http://REST.GATEWAY.SA/api/SendSMS"
    
    # Define the parameters for the SMS request
    params = {
        "api_id": "API71789973116",
        "api_password": "salik2023CS",
        "sms_type": "T",
        "encoding": "T",
        "sender_id": "Gateway.sa",
        "phonenumber": "966558688926",
        "textmessage": f"A {className} has been detected at {cam} with accuracy {conf}. Image File URL {image_url} ",
       
    }
    
    # POST request to send the SMS
    response = requests.post(sms_gateway_url, params=params)





def main():

    Camera_location = "#Makkah Region."

    cap = cv2.VideoCapture(0)

    model = YOLO("best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness= 2,
        text_thickness= 2,
        text_scale= 1,
        text_color= Color.white(),
    )

    classNames = ["Pothole"]




    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms = True)[0]

        detections= sv.Detections.from_ultralytics(result)

        
        boxes = result.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            labels = [
                f"{classNames[cls]} {conf}"
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )
            # Check if a pothole is detected 
            if  conf >= 0.5:
                image_filename = "pothole_image.jpg"
                cv2.imwrite(image_filename, frame)  # Save the image to a file
                send_sms(Camera_location, image_filename, classNames[cls], conf )  



        cv2.imshow("YOLOv8", frame)





        if ( cv2.waitKey(30) == 27 ):
            break








if __name__ == "__main__":
    main()