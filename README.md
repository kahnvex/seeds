# seeds

Seed identification Kaggle challenge.

This training and inference script is best described via it's help
dialog and looking at the source code.

```
$ git clone git@github.com:kahnvex/seeds.git && cd seeds/
$ pip install -r requirements.txt
$ python3 seeds/seeds.py --help
usage: seeds.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--learning-rate LEARNING_RATE] [--save-path SAVE_PATH]
                [--base-model {xception,resnet,densenet}] [--test]
                [--load-model] [--img-size IMG_SIZE] [--outfile OUTFILE]
                --name NAME

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -e EPOCHS
  --batch-size BATCH_SIZE, -b BATCH_SIZE
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
  --save-path SAVE_PATH
                        Path in which to save the model's h5 file
  --base-model {xception,resnet,densenet}
                        Architecture hyperparameter
  --test                Classify the test set
  --load-model          Load the model from disk instead of training
  --img-size IMG_SIZE   Input image size
  --outfile OUTFILE     Name of file for results
  --name NAME           Name of the model
```
