import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO


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


def get_metrics(y_true, y_pred, task):
    auc = getAUC(y_true, y_pred, task)
    acc = getACC(y_true, y_pred, task)
    return auc, acc


def test(images, labels, index, ckpt_path, task):

    tflite_model, input_details, output_details = load_tflite(ckpt_path)

    y_true = np.zeros((images.shape[0], labels.shape[1]))
    y_score = np.zeros((images.shape[0], len(index)))

    for idx in tqdm(range(images.shape[0])):
        label = labels[idx]
        img = images[idx]
        
        score = test_single_img(img, tflite_model, input_details, output_details)[index]        
        score = np.expand_dims(score, axis=0)

        y_true[idx] = np.array([label])
        y_score[idx] = score

    auc, acc = get_metrics(y_true, y_score, task)

    return auc, acc


def main(flag, input_root, output_root, model_dir, model_id):

    flag_dict = {
        "pathmnist": 'PathMNIST',
        "chestmnist": 'ChestMNIST',
        "dermamnist": 'DermaMNIST',
        "octmnist": 'OCTMNIST',
        "pneumoniamnist": 'PneumoniaMNIST',
        "retinamnist": 'RetinaMNIST',
        "breastmnist": 'BreastMNIST',
        "organamnist": 'OrganAMNIST',
        "organcmnist": 'OrganCMNIST',
        "organsmnist": 'OrganSMNIST',
        "bloodmnist": 'BloodMNIST',
        "tissuemnist": 'TissueMNIST'}


    dataroot = os.path.join(input_root, '%s.npz' % (flag))
    npz_file = np.load(dataroot)

    train_img = npz_file['train_images']
    train_label = npz_file['train_labels']
    val_img = npz_file['val_images']
    val_label = npz_file['val_labels']
    test_img = npz_file['test_images']
    test_label = npz_file['test_labels']

    label_dict_path = os.path.join(model_dir, flag, '%s_model%s' % (flag, model_id), '%s_model%s_dict.txt' % (flag_dict[flag], model_id))
    model_path = os.path.join(model_dir, flag, '%s_model%s' % (flag, model_id), '%s_model%s.tflite' % (flag_dict[flag], model_id))

    idx = []
    labels = INFO[flag]['label']
    task = INFO[flag]['task']
    with open(label_dict_path) as f:
        line = f.readline()
        while line:
            idx.append(line[:-1].lower())
            line = f.readline()

    index = []
    for key in labels:
        index.append(idx.index(labels[key].lower()))

    print('train:')
    train_auc, train_acc = test(train_img, train_label, index, model_path, task)
    print('val:')
    val_auc, val_acc = test(val_img, val_label, index, model_path, task)
    print('test:')
    test_auc, test_acc = test(test_img, test_label, index, model_path, task)

    log = '%s%s\n' % (flag, model_id)
    train_log = 'train  auc: %.5f  acc: %.5f \n' % (train_auc, train_acc)
    val_log = 'val  auc: %.5f  acc: %.5f \n' % (val_auc, val_acc)
    test_log = 'test  auc: %.5f  acc: %.5f \n' % (test_auc, test_acc)

    log = log + train_log + val_log + test_log + '\n'
    print(log)

    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    with open(os.path.join(output_root, '%s.txt' % (flag)), 'a') as f:
        f.write(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        type=str)
    parser.add_argument('--output_root',
                        default='./automl_vision_results',
                        type=str)
    parser.add_argument('--model_dir',
                        default='/data/MedMNIST_models/automl_vision',
                        type=str)
    parser.add_argument('--model_id',
                        default='1',
                        type=str)

    args = parser.parse_args()
    flag = args.flag
    input_root = args.input_root
    model_dir = args.model_dir
    model_id = args.model_id
    output_root = args.output_root

    main(flag, input_root, output_root, model_dir, model_id)
