import argparse
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import tensorflow as tf

import medmnist
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, input_root, output_root, model_dir, run):

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    _ = getattr(medmnist, INFO[data_flag]['python_class'])(
            split="train", root=input_root, download=True)
    
    dataroot = os.path.join(input_root, '%s.npz' % (data_flag))
    npz_file = np.load(dataroot)

    train_img = npz_file['train_images']
    train_label = npz_file['train_labels']
    val_img = npz_file['val_images']
    val_label = npz_file['val_labels']
    test_img = npz_file['test_images']
    test_label = npz_file['test_labels']

    label_dict_path = glob(os.path.join(model_dir, '*.txt'))[0]
    
    model_path = glob(os.path.join(model_dir, '*.tflite'))[0]
    
    idx = []
    labels = INFO[data_flag]['label']
    task = INFO[data_flag]['task']
    with open(label_dict_path) as f:
        line = f.readline()
        while line:
            idx.append(line[:-1].lower())
            line = f.readline()

    index = []
    for key in labels:
        index.append(idx.index(labels[key].lower()))

    test(data_flag, train_img, index, model_path, 'train', output_root, run)
    test(data_flag, val_img, index, model_path, 'val', output_root, run)
    test(data_flag, test_img, index, model_path, 'test', output_root, run)


def load_tflite(ckpt_path):
    tflite_model = tf.lite.Interpreter(model_path=ckpt_path)
    tflite_model.allocate_tensors()
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    return tflite_model, input_details, output_details


def test_single_img(img, tflite_model, input_details, output_details):
    img = Image.fromarray(np.uint8(img)).resize((224, 224)).convert('RGB')
    img = np.expand_dims(np.array(img), axis=0)
    
    tflite_model.set_tensor(input_details[0]['index'], img)
    tflite_model.invoke()
    score = tflite_model.get_tensor(output_details[0]['index'])
    score = np.array(score).squeeze().astype(np.float32) / 255
    return score


def get_key(dic, val):
    for key in dic:
        if dic[key] == val:
            return key


def test(data_flag, images, index, model_path, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)

    tflite_model, input_details, output_details = load_tflite(model_path)

    y_score = np.zeros((images.shape[0], len(index)))

    for idx in tqdm(range(images.shape[0])):
        img = images[idx]    
        score = test_single_img(img, tflite_model, input_details, output_details)[index]        
        score = np.expand_dims(score, axis=0)
        y_score[idx] = score

    auc, acc = evaluator.evaluate(y_score, output_root, run)
    print('%s  auc: %.5f  acc: %.5f' % (split, auc, acc))

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
                        default='./automl_vision',
                        type=str)
    parser.add_argument('--model_path',
                        default='./MedMNIST_models/pathmnist/automl_vision_1',
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    input_root = args.input_root
    model_path = args.model_path
    output_root = args.output_root
    run = args.run

    main(data_flag, input_root, output_root, model_path, run)
