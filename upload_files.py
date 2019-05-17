import subprocess
import time
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from google.cloud import storage



args = ['nohup', 'python', '-u', '/Users/polaras/PycharmProjects/Speech_recognition_Project/train.py',
        '>', 'log.txt', '&']

start_vm_speech = ['/Users/polaras/Documents/google-cloud-sdk/bin/gcloud', 'compute', 'instances', 'start',
                   'dt2119-speaker-verification-vm']

stop_vm_speech = ['gcloud', 'compute', 'instances', 'stop', 'dt2119-speaker-verification-vm']

copy_file_speech = ['/Users/polaras/Documents/google-cloud-sdk/bin/gcloud', 'compute', 'scp', '--project',
                    'dt2119-speaker-verification-vm',
                    '--zone', 'europe-west1-b', '--recurse',
                    '/Users/polaras/Downloads/termpapersVT2017', 'polaras@test-instance:/home/polaras/']

start_vm_test = ['/Users/polaras/Documents/google-cloud-sdk/bin/gcloud', 'compute', 'instances', 'start',
                 'test-instance']
stop_vm_test = ['/Users/polaras/Documents/google-cloud-sdk/bin/gcloud', 'compute', 'instances', 'stop', 'test-instance']

copy_file_test = ['/Users/polaras/Documents/google-cloud-sdk/bin/gcloud', 'compute', 'scp', '--project',
                  'test-instance',
                  '--zone', 'us-east1-b', '--recurse',
                  '/Users/polaras/Downloads/termpapersVT2017', 'polaras@test-instance:/home/polaras/']


def main():
    subprocess.Popen(start_vm_test)
    time.sleep(100)
    subprocess.Popen(copy_file_test)
    subprocess.Popen(stop_vm_test)


# subprocess.Popen(start_vm_speech)
# subprocess.Popen(copy_file_speech)
# subprocess.Popen(stop_vm_speech)

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



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


if __name__ == "__main__":
    # main()

    credentials = GoogleCredentials.get_application_default()
    compute = discovery.build('compute', 'v1', credentials=credentials)

    start_request = compute.instances().start(
        project='dt2119-speaker-verification',
        zone='us-east1-b',
        instance='test-instance'
    )
    start_response = start_request.execute()

    wait_for_operation(
        compute=compute,
        project='dt2119-speaker-verification',
        zone='us-east1-b',
        operation=start_response['name']
    )



    # subprocess.Popen(copy_file_test)


    stop_request = compute.instances().stop(
        project='dt2119-speaker-verification',
        zone='us-east1-b',
        instance='test-instance'
    )

    stop_response = stop_request.execute()

    wait_for_operation(
        compute=compute,
        project='dt2119-speaker-verification',
        zone='us-east1-b',
        operation=stop_response['name']
    )



