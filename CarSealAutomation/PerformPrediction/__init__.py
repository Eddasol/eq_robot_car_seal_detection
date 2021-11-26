import logging
import os
import dotenv
import azure.functions as func
import json
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    __version__,
)
import numpy as np
import cv2
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)

from msrest.authentication import ApiKeyCredentials

PREDICTION_THRESHOLD = 0.5


def connect_to_container():
    dotenv.load_dotenv()
    blob_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client = BlobServiceClient.from_connection_string(
        blob_connection_string
    )
    return blob_service_client


def read_file_from_blob(blob_service_client, file_path):
    container_name_read = "mission-reports"
    blob_client_read = blob_service_client.get_blob_client(
        container=container_name_read, blob=file_path
    )
    blob_data = blob_client_read.download_blob().readall()
    return blob_data


def write_file_to_blob(blob_service_client, file_path, data):
    container_name_write = "analysis-results"
    blob_client_write = blob_service_client.get_blob_client(
        container=container_name_write, blob=file_path
    )
    blob_client_write.upload_blob(data)


def analyse_image(predict_images, image):
    predictions = predict_images.predict_image(image=image)
    image = predict_images.draw_prediction(image, predictions)
    return image, predictions


class PredictImages:
    def __init__(self):
        TRAINING_KEY = os.environ["TRAINING_KEY"]
        TRAINING_ENDPOINT = os.environ["TRAINING_ENDPOINT"]
        PREDICTION_KEY = os.environ["PREDICTION_KEY"]
        PREDICTION_ENDPOINT = os.environ["PREDICTION_ENDPOINT"]
        # Authenticate the training client
        credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
        trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

        # Authenticate the prediction client
        prediction_credentials = ApiKeyCredentials(
            in_headers={"Prediction-key": PREDICTION_KEY}
        )
        self.predictor = CustomVisionPredictionClient(
            PREDICTION_ENDPOINT, prediction_credentials
        )

        project_name = "car_seal_train_and_validation"
        self.publish_iteration_name = "Iteration3"
        self.max_byte_size = 4000000

        projects = trainer.get_projects()
        project_id = next((p.id for p in projects if p.name == project_name), None)

        print("Connecting to existing project...")
        self.project = trainer.get_project(project_id)

    @staticmethod
    def draw_prediction(image, predictions):
        GREEN = (0, 255, 0)
        BBOX_LINE_SIZE = 5
        nparr = np.fromstring(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_height = image.shape[0]
        img_width = image.shape[1]

        for prediction in predictions:
            if prediction.probability < PREDICTION_THRESHOLD:
                continue
            print(
                "\t",
                prediction.tag_name,
                ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(
                    prediction.probability * 100,
                    prediction.bounding_box.left,
                    prediction.bounding_box.top,
                    prediction.bounding_box.width,
                    prediction.bounding_box.height,
                ),
            )

            left = prediction.bounding_box.left * img_width
            top = prediction.bounding_box.top * img_height
            width = prediction.bounding_box.width * img_width
            height = prediction.bounding_box.height * img_height

            x0 = int(left)
            y0 = int(top)
            x1 = int(left + width)
            y1 = int(top + height)
            cv2.rectangle(image, (x0, y0), (x1, y1), GREEN, BBOX_LINE_SIZE)
        image = cv2.imencode(".jpg", image)[1].tostring()

        return image

    def predict_image(self, image: str):
        print(f"Predicting on image")

        # Send image and get back the prediction results
        results = self.predictor.detect_image(
            self.project.id, self.publish_iteration_name, image
        )
        return results.predictions


def main(myblob: func.InputStream, inputblob: bytes):
    if myblob.name.split("_")[-1] == "NAVI.json":
        navi_metadata = json.loads(inputblob)

        # Initialize azure connections
        blob_service_client = connect_to_container()
        predict_images = PredictImages()

        for image_metadata in navi_metadata:
            tag = image_metadata["tag"]
            file_name = image_metadata["file_name"]
            file_folders = myblob.name.split("/")[1:-1]
            file_path = "".join([folder + "/" for folder in file_folders]) + file_name

            image = read_file_from_blob(blob_service_client, file_path)
            logging.info(f"\n\nImage_type: {type(image)}")
            analysed_image, predictions = analyse_image(predict_images, image)

            file_path_split = file_path.split(".")
            file_path_analysed = file_path_split[0] + "_carseal." + file_path_split[1]

            write_file_to_blob(blob_service_client, file_path_analysed, analysed_image)
            formated_pred = []
            for prediction in predictions:
                if prediction.probability < PREDICTION_THRESHOLD:
                    continue
                pred = {
                    "label": prediction.tag_name,
                    "probability": prediction.probability,
                    "bounding_box": {
                        "left": prediction.bounding_box.left,
                        "top": prediction.bounding_box.top,
                        "width": prediction.bounding_box.width,
                        "height": prediction.bounding_box.height,
                    },
                }
                formated_pred.append(pred)

            img_metadata = {
                "tag": tag,
                "analysed_image_file": file_path_analysed.split("/")[-1],
                "original_file": file_name,
                "timestamp": image_metadata["timestamp"],
                "position": image_metadata["position"],
                "orientation": image_metadata["orientation"],
                "detection": formated_pred,
            }
            metadata_json = json.dumps(img_metadata).encode("utf-8")
            file_path_analysed_json = file_path_split[0] + "_carseal" + ".json"
            write_file_to_blob(
                blob_service_client, file_path_analysed_json, metadata_json
            )


# PSUDO
# If metadata      ->   load in new area, unchanged
# If navi_metadata ->   load in new area, unchanged?
#                       go trough tags and see what analazis to do
#                       load image, do analisys and upload analyzed image
