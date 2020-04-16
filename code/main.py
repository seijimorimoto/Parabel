from parabel import Parabel
from utils import load_dataset, prepare_nltk, vectorize_documents, ValidationMetric


def main():
    prepare_nltk()
    dataset = load_dataset('data/econbiz/econbiz.csv')
    X = dataset['title']
    Y = dataset['labels']
    folds = dataset['fold']
    parabel = Parabel()
    X = vectorize_documents(X)
    metrics = [ValidationMetric.F1_score_sample, ValidationMetric.Precision_at_k, ValidationMetric.Precision_at_k, ValidationMetric.Precision_at_k]
    metrics_args = [None, 1, 3, 5]
    scores = parabel.cross_validate(
        X, Y, folds, convert_X=False, metrics=metrics, metrics_args=metrics_args)
    print(f'F1_score_sample: {scores[0]}')
    print(f'Precision_at_1: {scores[1]}')
    print(f'Precision_at_3: {scores[2]}')
    print(f'Precision_at_5: {scores[3]}')


if __name__ == "__main__":
    main()
