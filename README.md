# seeds

Seed identification Kaggle challenge.

This training and inference script is best described via its help
dialog and looking at the source code.

```
$ git clone git@github.com:kahnvex/seeds.git && cd seeds/
$ pip install -r requirements.txt
$ python3 -m seeds.seeds --help
usage: seeds.py [-h] --name NAME [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--learning-rate LEARNING_RATE] [--save-path SAVE_PATH]
                [--base-model {xception,resnet,densenet121,densenet169}]
                [--test] [--load-model] [--img-size IMG_SIZE]
                [--outfile OUTFILE]
                [--optimizer {adam,rmsprop,adagrad,adadelta}] [--pdist]
                [--ensemble ENSEMBLE [ENSEMBLE ...]]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of the model
  --epochs EPOCHS, -e EPOCHS
  --batch-size BATCH_SIZE, -b BATCH_SIZE
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
  --save-path SAVE_PATH
                        Path in which to save the model's h5 file
  --base-model {xception,resnet,densenet121,densenet169}
                        Architecture hyperparameter
  --test                Classify the test set
  --load-model          Load the model from disk instead of training
  --img-size IMG_SIZE   Input image size
  --outfile OUTFILE     Name of file for results
  --optimizer {adam,rmsprop,adagrad,adadelta}
                        The optimization function to use
  --pdist               Output the probability distribution as a CSV
  --ensemble ENSEMBLE [ENSEMBLE ...]
                        When testing, ensembe 1+ models using mean
```

You can see the Kaggle seed identification challenge [here](https://www.kaggle.com/c/plant-seedlings-classification).

As of the time of this writing, several models trained by this script
have scored in the top 5% of the Kaggle Seedling identification challenge.
