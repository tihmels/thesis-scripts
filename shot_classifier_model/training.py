#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import json
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=lambda p: Path(p).resolve(strict=True),
                    default="/Users/tihmels/TS/ts-dataset")
parser.add_argument('-c', '--classes', type=str, default=[], nargs='+')
parser.add_argument('-e', '--epochs', type=int, default=15, help="number of epochs")
parser.add_argument('--bs', '--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--shape', type=lambda s: list(map(int, s.split('x'))), default="331x331", help="<width>x<height>")
parser.add_argument('--ft', '--finetune', type=int, default=0)


def generators(path, classes, target_size, batch_size, preprocessing):
    img_data_gen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        horizontal_flip=True,
        validation_split=0.2,
    )

    training_dataset = img_data_gen.flow_from_directory(
        path,
        seed=10,
        target_size=target_size,
        classes=classes,
        batch_size=batch_size,
        subset='training'
    )

    validation_dataset = img_data_gen.flow_from_directory(
        path,
        seed=10,
        target_size=target_size,
        classes=classes,
        batch_size=batch_size,
        subset='validation'
    )

    return training_dataset, validation_dataset


def create_model(input_shape, n_classes, optimizer, fine_tune=0):
    conv_base = tf.keras.applications.InceptionResNetV2(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False)

    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    top_model = conv_base.output
    top_model = tf.keras.layers.Flatten(name='flatten')(top_model)
    top_model = tf.keras.layers.Dense(4096, activation='relu')(top_model)
    top_model = tf.keras.layers.Dense(1027, activation='relu')(top_model)
    top_model = tf.keras.layers.Dropout(0.2)(top_model)

    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def main(args):
    base_dir = Path(args.dataset)

    subdirs = sorted(list([d.name for d in base_dir.glob("*") if d.is_dir()]))

    classes = args.classes if len(args.classes) > 0 else subdirs

    assert all(
        clazz in subdirs for clazz in classes), f'each class need to be present as subdirectory in {base_dir}'

    train_ds, val_ds = generators(base_dir, classes, args.shape, args.bs,
                                  tf.keras.applications.inception_resnet_v2.preprocess_input)

    width, height = args.shape

    input_shape = (height, width, 3)

    optim_1 = tf.keras.optimizers.Adam(learning_rate=0.001)
    optim_2 = tf.keras.optimizers.Adam(lr=0.0001)

    model = create_model(input_shape, len(classes), optim_1 if args.ft == 0 else optim_2, args.ft)

    checkpoint = ModelCheckpoint(
        filepath=Path(Path(__file__).resolve().parent, 'model', 'ts_anchor_v1.weights.best.hdf5'),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=5,
                               restore_best_weights=True)

    model.fit(train_ds,
                     epochs=args.epochs,
                     validation_data=val_ds,
                     callbacks=[checkpoint, early_stop]
                     )

    model.save(
        Path(Path(__file__).resolve().parent, 'model', 'ts_anchor_model'),
    )

    with open(Path(Path(__file__).resolve().parent, 'model', 'classes.txt'), 'w') as file:
        json.dump(classes, file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
