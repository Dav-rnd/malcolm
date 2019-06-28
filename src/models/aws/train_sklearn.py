import argparse
import pandas as pd
import os

from sklearn.externals import joblib


def model_fn(model_dir):
    """
    Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def get_sklearn_class(s_import):
    """ Import the sklear classifier, e.g. from sklearn.ensemble import RandomForestClassifier
    Then return the imported class """
    print('Importing model: {}'.format(s_import))
    exec(s_import)
    return eval(s_import.split(' ')[-1])


def get_classifier(s_import, hyperparameters):
    clf_class = get_sklearn_class(s_import)
    print('Initializing model. Hyperparameters: {}'.format(hyperparameters))
    return clf_class(**hyperparameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: note that validation set is not used here

    # Hyperparameters and classifier name are described here
    parser.add_argument('--import', dest='s_import', type=str, required=True)
    parser.add_argument('--hyperparameters', dest='hyperparameters', type=str, required=True)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.ix[:, 0]
    train_X = train_data.ix[:, 1:]

    # Now use scikit-learn's classifier to train the model.
    clf = get_classifier(args.s_import, eval(args.hyperparameters))
    print('Training...')
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    print('Saving...')
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
