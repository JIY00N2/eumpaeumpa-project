from google.cloud import storage
import os

class Bucket_processor():
    def __init__(self, auth_key_path, project_id, bucket_name):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = auth_key_path

        self.project_id = project_id  # "mlops-348504"
        self.bucket_name = bucket_name # sm_mlops_data

    def download_from_bucket(self, bucket_data_path, local_save_path):
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)

        blob = bucket.blob(bucket_data_path)
        blob.download_to_filename(local_save_path)
        #blob = bucket.blob("capston_data/text/train/beauty_health.json")

        #blob.download_to_filename(ABS_PATH + "/test/beauty_health.json")

    def upload_to_bucket(self, bucket_save_path, local_data_path):
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)

        blob = bucket.blob(bucket_save_path)
        blob.upload_from_filename(local_data_path)