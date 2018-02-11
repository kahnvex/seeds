import argparse
import csv
import numpy as np
import os

from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from PIL import Image

from ensemble import MeanEnsemble


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True,
                        help='Name of the model')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=0.0001)
    parser.add_argument('--save-path', type=str, default='models/',
                        help='Path in which to save the model\'s h5 file')
    parser.add_argument('--base-model', type=str, default='xception',
                        choices=('xception', 'resnet', 'densenet121',
                                 'densenet169'),
                        help='Architecture hyperparameter')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Classify the test set')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Load the model from disk instead of training')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--outfile', type=str, default='results.csv',
                        help='Name of file for results')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=('adam', 'rmsprop', 'adagrad', 'adadelta'),
                        help='The optimization function to use')
    parser.add_argument('--pdist', action='store_true', default=False,
                        help='Output the probability distribution as a CSV')
    parser.add_argument('--ensemble', nargs='+', default=[],
                        help='When testing, ensembe 1+ models using mean')

    return parser.parse_args()


def get_base_model(args):
    if args.base_model == 'xception':
        return Xception
    elif args.base_model == 'resnet':
        return ResNet50
    elif args.base_model == 'densenet121':
        return DenseNet121
    elif args.base_model == 'densenet169':
        return DenseNet169


def get_optimizer(args):
    if args.optimizer == 'adam':
        return optimizers.Adam(lr=args.learning_rate)

    elif args.optimizer == 'rmsprop':
        return optimizers.RMSprop(lr=args.learning_rate)

    elif args.optimizer == 'adagrad':
        return optimizers.Adagrad()

    elif args.optimizer == 'adadelta':
        return optimizers.Adadelta()


def get_model(args, n_classes):
    insize = args.img_size
    BaseModel = get_base_model(args)
    base_model = BaseModel(weights='imagenet', include_top=False,
                           pooling='avg', input_shape=(insize, insize, 3))

    x = base_model.output

    if args.base_model == 'densenet169' or args.base_model == 'densenet121':
        x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, output=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=get_optimizer(args),
                  metrics=['accuracy'])
    return model


def train(args):
    n_classes = len(os.listdir('data/train'))
    img_size = (args.img_size, args.img_size)
    save_path = args.save_path
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    weights_path = '%s/%s.h5' % (save_path, args.name)
    os.makedirs(save_path, exist_ok=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=0)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                 save_best_only=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                                  verbose=0, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0)
    tensorboard = TensorBoard(write_grads=True, log_dir='logs/%s' % args.name)
    callbacks = [early_stop, checkpoint, reduce_lr, tensorboard]

    model = get_model(args, n_classes)
    train = datagen.flow_from_directory('data/train', target_size=img_size,
                                        batch_size=args.batch_size)
    vald = datagen.flow_from_directory('data/validation',
                                       target_size=img_size,
                                       batch_size=args.batch_size)

    model.fit_generator(train, validation_data=vald, epochs=args.epochs,
                        callbacks=callbacks)

    return model


def load_pretrained(args):
    if not args.ensemble:
        path = os.path.normpath('%s/%s.h5' % (args.save_path, args.name))
        return load_model(path)

    models = []

    for modelname in args.ensemble:
        path = os.path.normpath('%s/%s.h5' % (args.save_path, modelname))
        models.append(load_model(path))

    return MeanEnsemble(models)


def test(args, model):
    testpath = 'data/original/test'
    classes = sorted(os.listdir('data/train'))
    os.makedirs('results/', exist_ok=True)

    with open('results/%s.csv' % args.name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'species'])

        for imgpath in os.listdir(testpath):
            img = Image.open('%s/%s' % (testpath, imgpath))
            img = img.resize((args.img_size, args.img_size))
            img = (1./255) * np.array(img)
            pred = model.predict(np.array([img]))[0]
            classname = classes[np.argmax(pred)]

            writer.writerow([imgpath, classname])

    if args.pdist:
        with open('results/%s_pdist.csv' % args.name, 'w') as f:
            writer = csv.writer(f)
            header = ['img'] + classes
            writer.writerow(header)

            for imgpath in os.listdir(testpath):
                img = Image.open('%s/%s' % (testpath, imgpath))
                img = img.resize((args.img_size, args.img_size))
                img = (1./255) * np.array(img)
                pred = model.predict(np.array([img]))[0]

                writer.writerow([imgpath] + [str(e) for e in pred.tolist()])


def main(args):
    if not args.load_model:
        model = train(args)
    else:
        model = load_pretrained(args)

    if args.test:
        test(args, model)


if __name__ == '__main__':
    main(get_args())
