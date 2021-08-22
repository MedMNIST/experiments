import os
import argparse
import time
import numpy as np
import tensorflow as tf
import autokeras as ak
import kerastuner
from tensorflow.keras.models import load_model

import medmnist
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, num_trials, input_root, output_root, gpu_ids, run, model_path):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids)

    info = INFO[data_flag]
    task = info['task']
    _ = getattr(medmnist, INFO[data_flag]['python_class'])(
            split="train", root=input_root, download=True)

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))

    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val = npz_file['val_images']
    y_val = npz_file['val_labels']
    x_test = npz_file['test_images']
    y_test = npz_file['test_labels']

    if model_path is not None:
        model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        test(model, data_flag, x_train, 'train', output_root, run)
        test(model, data_flag, x_val, 'val', output_root, run)
        test(model, data_flag, x_test, 'test', output_root, run)
        
    if num_trials == 0:
        return

    model = train(data_flag, x_train, y_train, x_val, y_val, task, num_trials, output_root, run)

    test(model, data_flag, x_train, 'train', output_root, run)
    test(model, data_flag, x_val, 'val', output_root, run)
    test(model, data_flag, x_test, 'test', output_root, run)


def train(data_flag, x_train, y_train, x_val, y_val, task, num_trials, output_root, run):

    clf = ak.ImageClassifier(
        multi_label=task=='multi-label, binary-class',
        project_name=data_flag,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        metrics=['AUC', 'accuracy'],
        objective=kerastuner.Objective("val_auc", direction="max"),
        overwrite=True,
        max_trials=num_trials
    )

    clf.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20
    )

    model = clf.export_model()

    try:
        model.save(os.path.join(output_root, '%s_autokeras_%s' % (data_flag, run)), save_format="tf")
    except Exception:
        model.save(os.path.join(output_root, '%s_autokeras_%s.h5' % (data_flag, run)))

    return model


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_score = model.predict(x)
    auc, acc = evaluator.evaluate(y_score, output_root, run)
    print('%s  auc: %.5f  acc: %.5f ' % (split, auc, acc))

    return auc, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--input_root',
                        default=DEFAULT_ROOT,
                        type=str)
    parser.add_argument('--output_root',
                        default='./autokeras',
                        type=str)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--num_trials',
                        default=20,
                        help='max_trials of autokeras search space, the script would only test model if num_trials=0',
                        type=int)

    args = parser.parse_args()
    data_flag = args.data_flag
    input_root = args.input_root
    output_root = args.output_root
    gpu_ids = args.gpu_ids
    run = args.run
    model_path = args.model_path
    num_trials = args.num_trials

    main(data_flag, num_trials, input_root, output_root, gpu_ids, run, model_path)
