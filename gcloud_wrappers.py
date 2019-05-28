import time
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

project = 'dt2119-speaker-verification'
zone = 'europe-west1-b'
instance = 'dt2119-speaker-verification-vm'


def start_speech_vm(compute=None):
    if compute is None:
        credentials = GoogleCredentials.get_application_default()

        compute = discovery.build(
            'compute',
            'v1',
            credentials=credentials
        )

    start_request = compute.instances().start(
        project=project,
        zone=zone,
        instance=instance
    )
    start_response = start_request.execute()

    wait_for_operation(
        compute=compute,
        project=project,
        zone=zone,
        operation=start_response['name']
    )


def stop_speech_vm(compute=None):
    if compute is None:
        credentials = GoogleCredentials.get_application_default()

        compute = discovery.build(
            'compute',
            'v1',
            credentials=credentials
        )

    stop_request = compute.instances().stop(
        project=project,
        zone=zone,
        instance=instance
    )

    stop_response = stop_request.execute()

    wait_for_operation(
        compute=compute,
        project=project,
        zone=zone,
        operation=stop_response['name']
    )


def wait_for_operation(compute, project, zone, operation):
    print('Waiting for operation to finish...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)


# from google.cloud import storage
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(source_file_name)
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))


if __name__ == "__main__":
    start_speech_vm()
    # stop_speech_vm(compute)
    pass
