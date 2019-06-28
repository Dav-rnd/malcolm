# MALCOLM - MAchine Learning COmmand Line Manager

This repo hosts a toolset aimed at symplifying data scientists' daily life, by automating a lot of tedious and repetitive tasks around dataset wrangling, training, testing, deploying etc.

It is currently focused on supervised learning tasks.

It notably unifies in a unique interface 3 famous ML libraries:

- Scikit Learn

- H2o

- AWS Sagemaker

Meaning that you can use seamlessly algorithms from these libraries, only by setting your training config file, leaving all the rest to MALCOLM.

Only two scripts are needed to do the job: one for pre-processing, one for training/testing/etc.

## Pre-processing Script

```
usage: python prepare_datasets.py

SageMaker pre-processing script

required arguments:
  -d DATASET_NAME    input dataset name (without extension)
  -t TARGET          feature name for the labels, or index if no headers

optional arguments:
  -h, --help           show this help message and exit
  -f DATA_FOLDER       data folder (default data)
  --infra-s3 S3_BUCKET configuration of the S3 bucket from infra_s3.yaml
  -p PREPROCESSOR      preprocessor name
  --scale              subtract the mean and divide by the std
  --encode             one-hot encoding of categorical variables (label encoding for H2O)
  --categ CAT1 CAT2    categorical features, encoded if --encoding is specified
  --headers            if enabled, attempt to parse headers
  --clean              ignore previously generated local files and erase intermediate files to regenerate them
  -v                   verbose mode
```

### Usage

Specify the dataset name with -d option, without file extension (e.g. `filename_20190319`).

The script needs a unique local working folder, that will be used to download/upload/store all files.

Specifying a S3 bucket + S3 folder will trigger storing the final results in the same S3 folder than where the input file can be downloaded.

The loading process works this way:

1. First it tries to load a local h5 file (from WORKING_FOLDER) for the input dataset.

2. (If not found) it tries to load a local csv file, then saves a h5 file for the raw data to speedup next launch.

3. (If not found) it tries to decompress a .csv.bz2 file, then performs step 2.

4. (If not found) it tries to download a .csv file from Amazon S3, using provided S3 credentials, then performs step 2.

5. (If not found) it tries to download a .csv.bz2 file from Amazon S3, using provided S3 credentials, then performs step 2.

6. (If not found) it raises a FileNotFoundError exception.

In order to upload the dataset to S3, your AWS credentials (`aws_access_key_id` and `aws_secret_access_key`) must be stored in `~/.aws/credentials`, as recommended in the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html). The `region` parameter should also be specified, and the specified S3 bucket must exist.
Note that the current parsing of the libsvm format by SageMaker implies that runtime errors will arise a test time if the last categorical variable contains a new value. This issue won't arise if you use `--new-categ ignore` or if your dataset contains at least one numerical feature.

### Example

```bash
python src/prepare_datasets.py -d abalone_multi -p abalone -t label --headers --scale --encode --infra-s3 s3_dev
```

## Training/testing/deployment Script

```
usage: python src/run.py

SageMaker training/testing/deployment script

required arguments:
  -a ACTIONS              actions to perform among: t(training), p(predict), e(evaluate), g(grid search),
                              d(deploy endpoint): deploy the model trained or specified by --awsmid on SageMaker,
                              r(remove endpoint): delete the endpoint corresponding to the specified --awsmid
  -d DATASET_NAME         input dataset name (without extension)
  -c CONFIG_FILENAME      config file name based on `config/model_<configfilename>.yaml`
  --model CONFIG_ID       model config with a matching in `config/model_<configfilename>.yaml`

optional arguments:
  -h, --help              show this help message and exit
  -f DATA_FOLDER          data folder (default data)
  -w WORKING_FOLDER       working folder (default jobs)
  --infra-s3 S3_BUCKET    configuration of the S3 bucket from infra_s3.yaml (required for AWS algos)
  --infra-sm SM_CONFIG    configuration of the SageMaker platform from infra_sm.yaml (required for AWS algos)
  --h2o H2O               ip:port of a running h2o instance (default localhost:54321)
  --model-id MODEL_ID     model ID of a previously trained model. Ignored if 't' is part of the actions
  --clean                 ignore previously generated local files and erase intermediate files to regenerate them
```

The script will first attempt to find the `.h5` preprocessed dataset locally. If it does not exist, the dataset will be retrieved from the S3 bucket specified in `config/config_model.yaml`.

### Launching an H2O instance

Training an H2O algorithm will automatically launch a local H2O server. However, you may want to use an existing H2O instance (local or remote).

To launch an h2o instance on a remote server:
* connect to ```REMOTESERVER```
* ```nohup java -Xmx20g -jar /mnt/sdb/h2o/h2o-3.24.0.2/h2o.jar -port 54342 -nthreads 20 >/mnt/sdb/h2o/logs/h2o.log 2>&1 &```
  * ```-Xmx20g``` 20g of ram used
  * ```-nthreads 20``` using 20 threads, all available threads if not specified

To check if the instance is running point a browser to ```REMOTESERVER:54342```.

**Warning**: same versions of h2o python package and h2o instance are needed, check [```requirements.txt```](requirements.txt)

### Example

```bash
python src/run.py -a tpe -d abalone_multi -c abalone --model sklearn_rf [--model-id MODEL_ID]
python src/run.py -a tpedr -d abalone_multi -c abalone --model aws_xgb --infra-s3 s3_dev --infra-sm sm_dev [--model-id MODEL_ID]
```

### Future Work

We encourage submitting pull requests for new features and inevitable bug-fixing.
Current leads for improvements are:

- Deployment/cleaning of sklearn and H2O models on SageMaker through -a d/r actions

- (Parallel) execution of several models on a given dataset (including summary at the end of the evaluations)
