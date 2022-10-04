#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

from pathlib import Path

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

batch_size = 32


def generators(path, shape, preprocessing):
    img_data_gen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        horizontal_flip=True,
        validation_split=0.2,
    )

    height, width = shape

    train_dataset = img_data_gen.flow_from_directory(
        path,
        seed=10,
        target_size=(height, width),
        classes=('anchor', 'non_anchor'),
        batch_size=batch_size,
        subset='training'
    )

    val_dataset = img_data_gen.flow_from_directory(
        path,
        seed=10,
        target_size=(height, width),
        classes=('anchor', 'non_anchor'),
        batch_size=batch_size,
        subset='validation'
    )

    return train_dataset, val_dataset


def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    base_model = tf.keras.applications.VGG19(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False)

    if fine_tune > 0:
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False

    top_model = base_model.output
    top_model = tf.keras.layers.Flatten(name='flatten')(top_model)
    top_model = tf.keras.layers.Dense(4096, activation='relu')(top_model)
    top_model = tf.keras.layers.Dense(1027, activation='relu')(top_model)
    top_model = tf.keras.layers.Dropout(0.2)(top_model)

    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


base_dir = Path("/Users/tihmels/TS/ts-anchorset")

training1_ds, validation1_ds = generators(base_dir, (224, 224), tf.keras.applications.vgg19.preprocess_input)

input_shape = (224, 224, 3)
n_classes = 2
n_epochs = 5

optim_1 = tf.keras.optimizers.Adam(learning_rate=0.001)
optim_2 = tf.keras.optimizers.Adam(lr=0.0001)

vgg_model = create_model(input_shape, n_classes, optim_1)

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

vgg_model.fit(training1_ds,
              epochs=n_epochs,
              validation_data=validation1_ds,
              callbacks=[tl_checkpoint_1, early_stop],
              verbose=1
              )

vgg_model.save(Path(Path(__file__).parent.resolve(), 'model', 'ts_anchorshot_model'))
