import os
import argparse
import time
import numpy as np
import tensorflow as tf
import autokeras as ak
import kerastuner

from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO


def main(flag, input_root, output_root, gpu_ids):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids)

    info = INFO[flag]
    task = info['task']

    output_root = os.path.join(output_root, flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(flag)))
    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val = npz_file['val_images']
    y_val = npz_file['val_labels']
    x_test = npz_file['test_images']
    y_test = npz_file['test_labels']


    clf = ak.ImageClassifier(
        multi_label=task=='multi-label, binary-class',
        project_name=flag,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        metrics=['AUC', 'accuracy'],
        objective=kerastuner.Objective("val_auc", direction="max"),
        overwrite=True,
        max_trials=20
    )

    clf.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20
    )

    model = clf.export_model()

    # train
    y_pred = model.predict(x_train)
    train_auc, train_acc = get_metrics(y_train, y_pred, task)
    # val
    y_pred = model.predict(x_val)
    val_auc, val_acc = get_metrics(y_val, y_pred, task)
    # test
    y_pred = model.predict(x_test)
    test_auc, test_acc = get_metrics(y_test, y_pred, task)

    log = '%s\n' % (flag)
    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_auc, train_acc)
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_auc, val_acc)
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_auc, test_acc)

    log = log + train_log + val_log + test_log
    print(log)

    with open(os.path.join(output_root, '%s_autokeras_log.txt' % (flag)), 'a') as f:
        f.write(log)

    try:
        model.save(os.path.join(output_root, '%s_autokeras' % (flag)), save_format="tf")
    except Exception:
        model.save(os.path.join(output_root, '%s_autokeras.h5' % (flag)))


def get_metrics(y_true, y_pred, task):
    auc = getAUC(y_true, y_pred, task)
    acc = getACC(y_true, y_pred, task)
    return auc, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        type=str)
    parser.add_argument('--output_root',
                        default='./autokeras_results',
                        type=str)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)

    args = parser.parse_args()
    flag = args.flag
    input_root = args.input_root
    output_root = args.output_root
    gpu_ids = args.gpu_ids

    main(flag, input_root, output_root, gpu_ids)

