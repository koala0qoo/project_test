from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO
import tensorflow as tf
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets import cam_inception

from object_detection import evaluator
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import object_detection_evaluation
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import cam_utils
from object_detection.core import standard_fields as fields

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 24, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 200,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'output_file', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'default_image_size', None, 'Default image size')

FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 764


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)

    ####################
    #  Get the label map #
    ####################
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels.txt')

    category_index = {}
    categories = []
    label_map = open(PATH_TO_LABELS, 'r', encoding='utf-8')
    for line in label_map:
        cat = {}
        id = line.strip().split(":")[0]
        name = line.strip().split(":")[1]
        cat['id'] = int(id)
        cat['name'] = name
        category_index[int(id)] = cat
        categories.append(cat)

    ####################
    #  Get train data #
    ####################

    filename_queue = tf.train.string_input_producer(
        [FLAGS.dataset_dir + '/pj_vehicle_validation_0000%d-of-00004.tfrecord' % i for i in range(0, 4)], )
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                       })

    image = features['image/encoded']
    label = features['image/class/label']

    graph = tf.Graph().as_default()
    ####################
    #  Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=NUM_CLASSES, is_training=False)

    if hasattr(network_fn, 'default_image_size'):
        image_size = network_fn.default_image_size
    else:
        image_size = FLAGS.default_image_size

    #####################################
    #  Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

    image_processed = tf.image.decode_jpeg(image, channels=3)
    image_processed = image_preprocessing_fn(image_processed, image_size, image_size)

    images_processed, images, labels = tf.train.batch(
        [image_processed, image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels = tf.squeeze(labels)

    ####################
    #  Define the model #
    #####################
    logits, end_points = network_fn(images_processed)

    checkpoint_path = FLAGS.checkpoint_path
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    saver.restore(sess, checkpoint_path)

    #evaluator = object_detection_evaluation.ObjectDetectionEvaluator(categories, matching_iou_threshold=0.5)
    iou = []

    with sess:

        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        for i in range(FLAGS.max_num_batches):
            images_, labels_, logits_, end_points_ = sess.run([images, labels, logits, end_points])

            for j in range(FLAGS.batch_size):
                idx = i * FLAGS.batch_size + j
                if j == 0:
                    image_ = images_[j]
                    image_ = Image.open(BytesIO(image_), 'r')
                    image_.save(os.path.join(FLAGS.output_file, 'test_{0}_label_{1}.jpg'.format(i, labels_[j])))
                    image_np = np.array(image_)

                    logit_value = logits_[j]
                    feature_maps_A = end_points_['features_A'][j]

                    softmax = np.exp(logit_value) / np.sum(np.exp(logit_value), axis=0)

                    n_top = 1
                    predictions = np.argsort(-logit_value)[:n_top]
                    scores = -np.sort(-softmax)[:n_top]

                    print(predictions)
                    print(scores)
                    print(labels_[j])

                    # 生成heatmap
                    cam_A = cam_utils.CAMmap(feature_maps_A, logit_value, n_top)
                    cam_B = cam_utils.CAMmap(feature_maps_B, logit_value, n_top)
                    for k in range(n_top):
                        fm_a = cam_A[:, :, k]
                        cam_A[:, :, k] = (fm_a - fm_a.min())/(fm_a.max() - fm_a.min())
                        fm_b = cam_B[:, :, k]
                        cam_B[:, :, k] = (fm_b - fm_b.min())/(fm_b.max() - fm_b.min())
                    cam = np.maximum(cam_A, cam_B)
                    im_height = image_np.shape[0]
                    im_width = image_np.shape[1]

                    # 保存heatmap
                    cam_resize = np.zeros((im_height, im_width, n_top))
                    for k in range(n_top):
                        heatmap_resize = Image.fromarray(cam[:, :, k]).resize((im_width, im_height), Image.BILINEAR)
                        cam_resize[:, :, k] = np.array(heatmap_resize)
                        heatmap = cam_utils.grey2rainbow(cam_resize[:, :, k] * 255)
                        heatmap = Image.fromarray(heatmap)
                        heatmap.save(os.path.join(FLAGS.output_file, 'test_{0}_heatmap_{1}.jpg'.format(i, k)))

                    # 生成bounding_boxes
                    threshold = 0.75
                    boxes = cam_utils.bounding_box(cam_resize, threshold)

                    # 输出检测结果
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxes,
                        predictions.astype(np.int32),
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=0.001,
                        line_thickness=5)
                    plt.imsave(os.path.join(FLAGS.output_file, 'test_{0}_output.jpg'.format(i)), image_np)

                    # 计算评价指标
                    annotations_dir = os.path.join(FLAGS.dataset_dir, 'test_data/annotations')
                    boxes_, classes_ = cam_utils.get_boxes(annotations_dir, i)
                    for k in range(boxes.shape[0]):
                        boxes[k, 0] = boxes[k, 0] * im_height
                        boxes[k, 1] = boxes[k, 1] * im_width
                        boxes[k, 2] = boxes[k, 2] * im_height
                        boxes[k, 3] = boxes[k, 3] * im_width

                    if predictions[0] == labels_[j]:
                        iou_ = np_box_ops.iou(boxes, boxes_)[0][0]
                        iou.append(iou_)

                    '''
                    result_dict = {}
                    result_dict[fields.InputDataFields.groundtruth_boxes] = boxes_
                    result_dict[fields.InputDataFields.groundtruth_classes] = classes_
                    result_dict[fields.DetectionResultFields.detection_boxes] = boxes
                    result_dict[fields.DetectionResultFields.detection_scores] = scores
                    result_dict[fields.DetectionResultFields.detection_classes] = classes
                    evaluator.add_single_ground_truth_image_info(image_id=i, groundtruth_dict=result_dict)
                    evaluator.add_single_detected_image_info(image_id=i, detections_dict=result_dict)

        metrics = evaluator.evaluate()
        for key in metrics:
            print(metrics[key])
        '''

        mean_iou = np.array(iou).mean()
        print(mean_iou)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()