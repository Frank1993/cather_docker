import argparse
import os
import signal
import subprocess
import sys
import time
import pika
import shutil
import glob
import getpass

sys.path.append(os.path.abspath('/home/job/dockers/quality_orientation_image_classifier/python_modules/'))
sys.path.append(os.path.abspath('/home/job/dockers/quality_orientation_image_classifier/python_modules/apollo_python_common/protobuf/'))
import apollo_python_common.proto_api as proto_api
from apollo_python_common.ml_pipeline.multi_threaded_predictor import RabbitMQProvider
import orbb_metadata_pb2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',     help='Not used', required=False, default=None)
    parser.add_argument('--dataDir',     help='Not used', required=False, default=None)
    parser.add_argument('--inputPath',   help='Path to the input messages', required=True, default=None)
    parser.add_argument('--imagesPath',  help='Path to the images', required=True, default=None)
    parser.add_argument('--batchName',   help='Name of the image batch', required=True, default=None)
    parser.add_argument('--outputDir',   help='Detection oputput path', required=True, default=None)
    parser.add_argument('--logDir',      help='Log file', required=False, default=None)
    parser.add_argument('--count',       help='Count', required=False, default=None)
    args = vars(parser.parse_args())

    print('Script arguments:')
    print('\t{}: {}'.format('inputPath', args['inputPath']))
    print('\t{}: {}'.format('imagesPath', args['imagesPath']))
    print('\t{}: {}'.format('batchName', args['batchName']))
    print('\t{}: {}'.format('outputDir', args['outputDir']))
    print('\t{}: {}'.format('count', args['count']))
    return args

def get_proto_for_path(path):
    return proto_api.get_new_image_proto("-1", -1, path, "US", 0, 0, True)

def is_cluster_run(images_path):
    return images_path.startswith('/philly/') or images_path.startswith('/hdfs/')

def edit_image_path(image_proto, images_path):
    if is_cluster_run(images_path):
        image_proto.metadata.image_path = os.path.join(images_path + '.zip@', os.path.basename(image_proto.metadata.image_path))
    else:
        image_proto.metadata.image_path = os.path.join(images_path, os.path.basename(image_proto.metadata.image_path))

def read_proto_messages(input_path):
    print('\nReading proto messages from {} ...'.format(input_path))
    image_set = proto_api.read_imageset_file(input_path)
    proto_images = [image for image in image_set.images]
    print('{} proto images loaded.'.format(len(proto_images)))
    return proto_images

def create_rabbit_mq_channel():
    params = pika.ConnectionParameters(
        host='localhost',
        port=5672,
        credentials=pika.credentials.PlainCredentials('guest', 'guest'))
    connection = pika.BlockingConnection(parameters=params)
    return connection.channel()

def kill_processes(processes):
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait()
        except ProcessLookupError:
            pass

def wait_queue(queues, processes_to_kill, freq=5):
    while True:
        channel = create_rabbit_mq_channel()
        q = 0
        try:
            for queue_name in queues:
                q += channel.queue_declare(queue=queue_name, durable=True, exclusive=False, passive=True).method.message_count
            print(q)
        except pika.exceptions.ChannelClosed:
            q = -1

        if q == 0:
            break
        else:
            for p in processes_to_kill:
                if p.poll() != None:
                    raise Exception()

        time.sleep(freq)

    time.sleep(10)
    kill_processes(processes_to_kill)

## Main function
if __name__== '__main__':
    args = parse_arguments()
    open_processes = []
    user_name = getpass.getuser()
    print(user_name)
    subprocess.check_call(['sudo', 'chown', '-R', user_name, '/var/lib/rabbitmq'])

    try:
        mq_rabbit_proc = subprocess.Popen(['rabbitmq-server', 'start'], preexec_fn=os.setsid)
        open_processes.append(mq_rabbit_proc)
        time.sleep(5)

        elasticsearch_proc = subprocess.Popen(['/bin/bash', '/elasticsearch-6.6.0/bin/elasticsearch'], preexec_fn=os.setsid)
        open_processes.append(elasticsearch_proc)
        time.sleep(15)

        proto_images = read_proto_messages(args['inputPath'])
        mq_provider = RabbitMQProvider("localhost", 5672, "guest", "guest")
        queue_name = "RAW_IMAGES"

        if args['count'] != None:
            proto_images = proto_images[:int(args['count'])]

        for img_proto in proto_images:
            edit_image_path(img_proto, args['imagesPath'] + args['batchName'])
            img_proto = proto_api.add_mandatory_sensor_data_to_proto(img_proto, -1, -1, 0, 0, 0, 0, 0, 'dummy')
            mq_provider.send_message(queue_name, img_proto.SerializeToString())

        quality_orientation_image_classifier_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/quality_orientation_image_classifier/', preexec_fn=os.setsid)
        open_processes.append(quality_orientation_image_classifier_proc)
        time.sleep(10)

        quality_orientation_image_selector_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/quality_orientation_image_selector/', preexec_fn=os.setsid)
        open_processes.append(quality_orientation_image_selector_proc)
        time.sleep(10)

        geo_image_selector_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/geo_image_selector/', preexec_fn=os.setsid)
        open_processes.append(geo_image_selector_proc)
        time.sleep(5)

        wait_queue(['RAW_IMAGES'], [quality_orientation_image_classifier_proc])
    
        traffic_signs_detector_yolo_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/traffic_signs_detector_yolo/', preexec_fn=os.setsid)
        open_processes.append(traffic_signs_detector_yolo_proc)
        time.sleep(5)
        wait_queue(['US_IMAGES', 'CLASSIFIED_IMAGES', 'GOOD_IMAGES'], [traffic_signs_detector_yolo_proc, quality_orientation_image_selector_proc, geo_image_selector_proc])

        traffic_signs_classifier_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/traffic_signs_classifier/', preexec_fn=os.setsid)
        open_processes.append(traffic_signs_classifier_proc)
        time.sleep(5)
        wait_queue(['RAW_DETECTED_TRAFFIC_SIGNS'], [traffic_signs_classifier_proc])

        signpost_components_detector_retinanet_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/signpost_components_detector_retinanet/', preexec_fn=os.setsid)
        open_processes.append(signpost_components_detector_retinanet_proc)
        time.sleep(5)
        wait_queue(['DETECTED_TRAFFIC_SIGNS'], [signpost_components_detector_retinanet_proc])

        signpost_ocr_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/signpost_ocr/', preexec_fn=os.setsid)
        open_processes.append(signpost_ocr_proc)
        time.sleep(5)
        wait_queue(['DETECTED_COMPONENTS'], [signpost_ocr_proc])

        signs_facing_classifier_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/signs_facing_classifier/', preexec_fn=os.setsid)
        open_processes.append(signs_facing_classifier_proc)
        time.sleep(10)

        traffic_signs_positioner_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/traffic_signs_positioner/', preexec_fn=os.setsid)
        open_processes.append(traffic_signs_positioner_proc)
        time.sleep(10)

        disk_serializer_proc = subprocess.Popen(['sh', './run.sh'], cwd='./dockers/disk_serializer/', preexec_fn=os.setsid)
        open_processes.append(disk_serializer_proc)
        time.sleep(5)

        wait_queue(['DETECTIONS_WITH_TEXT', 'CLASSIFIED_SIGNS_FACING'], [signs_facing_classifier_proc, traffic_signs_positioner_proc])

        print("Last wait ...")
        disk_serializer_proc.wait()
        kill_processes([mq_rabbit_proc, elasticsearch_proc])
        
    except:
        print("Exception!")
        kill_processes(open_processes)
        raise

    print("Checking output folder ...")
    output_files = glob.glob('/home/job/output/*')
    if len(output_files) != 2:
        raise Exception()
    else:
        for output_file in output_files:
           name_splits = os.path.splitext(os.path.basename(output_file))
           destination_path = os.path.join(args['outputDir'], args['batchName'] + name_splits[1])
           print('Copying {} to {}.'.format(output_file, destination_path))
           shutil.copy(output_file, destination_path)
