import os
import argparse
import joblib
import numpy as np
import autosklearn.classification

from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO


def main(flag, time, input_root, output_root):

    time = time * 60 * 60
    
    info = INFO[flag]
    task = info['task']
    
    output_root = os.path.join(output_root, flag)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(flag)))

    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val = npz_file['val_images']
    y_val = npz_file['val_labels']
    x_test = npz_file['test_images']
    y_test = npz_file['test_labels']

    size = x_train[0].size
    X_train = x_train.reshape(x_train.shape[0], size, )
    X_val = x_val.reshape(x_val.shape[0], size, )
    X_test = x_test.reshape(x_test.shape[0], size, )
    
    if task != 'multi-label, binary-class':
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(time),
        per_run_time_limit=int(time/10),
        tmp_folder='./tmp/autosklearn_classification_medmnist_tmp/%s' % (flag),
        output_folder='./tmp/autosklearn_classification_medmnist_out/%s' % (flag),
        ml_memory_limit=9216,
        ensemble_memory_limit=6144,
        n_jobs=4,
    )

    automl.fit(X_train, y_train, X_val, y_val)

    joblib.dump(automl, os.path.join(output_root, 'autosklearn_%s.m' % (flag)))

    # train
    prob = automl.predict_proba(X_train)
    train_auc, train_acc = get_metrics(y_train, prob, task)
    # val
    prob = automl.predict_proba(X_val)
    val_auc, val_acc = get_metrics(y_val, prob, task)
    # test
    prob = automl.predict_proba(X_test)
    test_auc, test_acc = get_metrics(y_test, prob, task)

    log = '%s\n' % (flag)
    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_auc, train_acc)
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_auc, val_acc)
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_auc, test_acc)
    
    log = log + train_log + val_log + test_log
    print(log)

    with open(os.path.join(output_root, '%s_autosklearn_log.txt' % (flag)), 'a') as f:
        f.write(log)


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
                        default='./autosklearn_results',
                        type=str)
    parser.add_argument('--time',
                        default=2,
                        type=int)

    args = parser.parse_args()
    flag = args.flag
    input_root = args.input_root
    output_root = args.output_root
    time = args.time

    main(flag, time, input_root, output_root)
    