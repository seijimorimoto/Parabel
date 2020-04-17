import argparse
import os
import time
from code.parabel import Parabel
from code.utils import *
from joblib import dump, load

def perform_iterative_evaluation(X, Y, folds, cross_validate, outdir):
    '''
    Iteratively perform validation of Parabel on subsets of the dataset given. Each iteration, a
    sub-dataset of different size will be used to perform the validation procedure.

    :param X: matrix with shape (N, M), where N is the number of inputs in the whole dataset and M
    is the number of features of each input.

    :param Y: an array of size N, where N is the number of data points in the whole dataset. Each
    position within the array has a string of labels (each one separated by a TAB character) that
    correspond to the labels with which the respective data point was tagged.

    :param folds: an array of size N, where N is the number of data points in the whole dataset.
    Each value within the array must indicate the fold to which the corresponding input data point
    belongs.

    :param cross_validate: whether to perform cross validation using each subset of the dataset
    given, or just perform a single training-evaluation procedure with each of them.

    :param outdir: the directory where information regarding time and scores will be outputted.
    '''
    # Mapping folds to data points.
    print('Mapping folds to data points...')
    start_time = time.time()
    folds_dict = get_folds_dictionary(folds)
    duration = time.time() - start_time
    print(f'Finished mapping folds in {duration} seconds.')

    # Initializing variables.
    inputs = []
    labels = []
    parabel = Parabel()
    metrics = [ValidationMetric.F1_score_sample, ValidationMetric.Precision_at_k,
        ValidationMetric.Precision_at_k, ValidationMetric.Precision_at_k]
    metrics_args = [None, {'k': 1}, {'k': 3}, {'k': 5} ]
    folds_temp = dict()

    # Get initial subset of the whole dataset and perform cross validation or simple training and
    # evaluation (using the first fold as evaluation) with it.
    if cross_validate:
        for fold in range(10):
            inputs += get_values_from_indices(X, folds_dict[fold])
            labels += get_values_from_indices(Y, folds_dict[fold])
            folds_temp[fold] = folds_dict[fold]
        parabel.cross_validate(inputs, labels, folds_temp,
            convert_X=False, metrics=metrics, metrics_args=metrics_args, outdir=outdir) 
    else:
        for fold in range(1, 10):
            inputs += get_values_from_indices(X, folds_dict[fold])
            labels += get_values_from_indices(Y, folds_dict[fold])
        test_inputs = get_values_from_indices(X, folds_dict[0])
        test_labels = get_values_from_indices(Y, folds_dict[0])
        parabel.train(inputs, labels, 100, convert_X=False, outdir=outdir)
        parabel.evaluate(test_inputs, test_labels, 10,
            metrics=metrics, metrics_args=metrics_args, outdir=outdir)

    # Iteratively double the size of the subset to use by adding more data from the original full
    # dataset. In each iteration, perform the cross validation procedure or simple training and
    # evaluation (using the first fold as evaluation).
    last_index = 0
    folds_temp[10] = []
    for i in range(3):
        dataset_size = len(inputs)
        new_indices = folds_dict[10][last_index : last_index + dataset_size]
        inputs += get_values_from_indices(X, new_indices)
        labels += get_values_from_indices(Y, new_indices)
        last_index = last_index + dataset_size
        folds_temp[10] += new_indices
        if cross_validate:
            parabel.cross_validate(inputs, labels, folds_temp,
                convert_X=False, metrics=metrics, metrics_args=metrics_args, outdir=outdir)
        else:
            parabel.train(inputs, labels, 100, convert_X=False, outdir=outdir)
            parabel.evaluate(test_inputs, test_labels, 10,
                metrics=metrics, metrics_args=metrics_args, outdir=outdir)
    
    # Now use the full original dataset and perform cross validation with it or simple training and
    # evaluation (using the first fold as evaluation).
    new_indices = folds_dict[10][last_index : ]
    inputs += get_values_from_indices(X, new_indices)
    labels += get_values_from_indices(Y, new_indices)
    folds_temp[10] += new_indices
    if cross_validate:
        parabel.cross_validate(inputs, labels, folds_temp,
            convert_X=False, metrics=metrics, metrics_args=metrics_args, outdir=outdir)
    else:
        parabel.train(inputs, labels, 100, convert_X=False, outdir=outdir)
        parabel.evaluate(test_inputs, test_labels, 10,
            metrics=metrics, metrics_args=metrics_args, outdir=outdir)


def main():
    '''
    Main method for evaluating the performance of the Parabel algorithm with different subsets of
    datasets.
    '''
    # Specifying and parsing command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Dataset to use')
    args = parser.parse_args()

    # Validating the command-line argument.
    if args.dataset not in ['econbiz', 'pubmed']:
        print('The only available datasets are \'econbiz\' and \'pubmed\'.')
    else:
        # Preparing environment and loading dataset.
        prepare_nltk()
        print('Loading dataset...')
        start_time = time.time()
        dataset = load_dataset(f'data/{args.dataset}.csv')
        print(f'Finished loading dataset in {time.time() - start_time} seconds.')
        Y = dataset['labels']
        folds = dataset['fold']

        # Vectorize the inputs in the dataset if no vector-version of them exists in disk. If one
        # does exist in disk, load it instead.
        if not os.path.exists(f'results/{args.dataset}/input_vectors.joblib'):
            print('Vectorizing inputs...')
            start_time = time.time()
            X = dataset['title']
            X = vectorize_documents(X)
            duration = time.time() - start_time
            print(f'Finished vectorizing inputs in {duration} seconds.')
            dump(X, f'results/{args.dataset}/input_vectors.joblib')
            save_keys_values_to_csv(
                f'results/{args.dataset}/times.csv', ['Vectorize inputs'], [duration])
        else:
            X = load(f'results/{args.dataset}/input_vectors.joblib')

        # Carry out the iterative evaluation procedure.
        perform_iterative_evaluation(X, Y, folds, False, f'results/{args.dataset}/')


if __name__ == "__main__":
    main()
