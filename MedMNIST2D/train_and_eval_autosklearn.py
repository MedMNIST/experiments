import os
import argparse
import joblib
import numpy as np
import autosklearn.classification

import medmnist
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, time, input_root, output_root, run, model_path):

    time = time * 60 * 60
    
    info = INFO[data_flag]
    task = info['task']
    _ = getattr(medmnist, INFO[data_flag]['python_class'])(
            split="train", root=input_root, download=True)
    
    output_root = os.path.join(output_root, data_flag)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))

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

    if model_path is not None:
        model = joblib.load(model_path)
        test(model, data_flag, X_train, 'train', output_root, run)
        test(model, data_flag, X_val, 'val', output_root, run)
        test(model, data_flag, X_test, 'test', output_root, run)

    if time == 0:
        return

    model = train(data_flag, time, X_train, y_train, X_val, y_val, run)
    
    test(model, data_flag, X_train, 'train', output_root, run)
    test(model, data_flag, X_val, 'val', output_root, run)
    test(model, data_flag, X_test, 'test', output_root, run)


def train(data_flag, time, X_train, y_train, X_val, y_val, run):

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(time),
        per_run_time_limit=int(time/10),
        tmp_folder='./tmp/autosklearn_classification_medmnist_tmp/%s' % (data_flag),
        output_folder='./tmp/autosklearn_classification_medmnist_out/%s' % (data_flag),
        ml_memory_limit=9216,
        ensemble_memory_limit=6144,
        n_jobs=4,
    )

    automl.fit(X_train, y_train, X_val, y_val)

    joblib.dump(automl, os.path.join(output_root, '%s_autosklearn_%s.m' % (data_flag, run)))

    return automl


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_score = model.predict_proba(x)
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
                        default='./autosklearn',
                        type=str)
    parser.add_argument('--time',
                        default=2,
                        help='run time (hours) for autokeras, the script will only test models if set time to 0',
                        type=int)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    input_root = args.input_root
    output_root = args.output_root
    time = args.time
    run = args.run
    model_path = args.model_path

    main(data_flag, time, input_root, output_root, run, model_path)
    