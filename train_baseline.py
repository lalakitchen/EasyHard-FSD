import os
import numpy as np
import argparse
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.utils import class_weight

from tensorflow.keras.applications import (
    EfficientNetV2S,
    ResNet50
)

# =========================
# IMPORT IMAGE GENERATOR
# =========================
from utils.ImageGenerator import build_dataset, AUG_BATCH

# =========================
# GPU SETUP
# =========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# =========================
# LABEL UTILS
# =========================
def class_map(category):
    label = np.zeros(2, dtype=np.float32)
    label[0 if category == 'Live' else 1] = 1
    return label

# =========================
# DATA LOADER (NUMPY ONLY)
# =========================
def load_data(paths):
    imgs, labels, cats = [], [], []
    for p in paths:
        try:
            cat = p.split('/')[-3]
            img = cv2.imread(p)
            img = cv2.resize(img, (256, 256))
            imgs.append(img)
            labels.append(class_map(cat))
            cats.append(cat)
        except:
            print("Error:", p)
    return np.array(imgs), np.array(labels), np.array(cats)

# =========================
# MODEL FACTORY
# =========================
def build_model(model_name, input_shape=(256,256,3), num_classes=2):

    if model_name == 'resnet':
        base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )

    elif model_name == 'effnet':
        base = EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )

    elif model_name == 'vit':
        inputs = layers.Input(shape=input_shape)
        x = layers.Rescaling(1./255)(inputs)

        patch_size = 16
        num_patches = (256 // patch_size) ** 2
        projection_dim = 64

        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [-1, num_patches, patch_size*patch_size*3])

        x = layers.Dense(projection_dim)(patches)
        x = layers.LayerNormalization()(x)

        for _ in range(4):
            attn = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=projection_dim
            )(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)

            mlp = layers.Dense(projection_dim*2, activation='gelu')(x)
            mlp = layers.Dense(projection_dim)(mlp)
            x = layers.Add()([x, mlp])
            x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    else:
        raise ValueError("Model must be one of: resnet | effnet | vit")

    x = layers.Input(shape=input_shape)
    y = base(x)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(num_classes, activation='softmax')(y)

    return tf.keras.Model(x, y)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2015')
    parser.add_argument('--scanner', default='CrossMatch')
    parser.add_argument('--model', required=True,
                        choices=['resnet', 'effnet', 'vit'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--exp_name', required=True)
    args = parser.parse_args()

    # =========================
    # SAFETY CHECK
    # =========================
    assert args.batch_size == AUG_BATCH, \
        f"batch_size must equal AUG_BATCH ({AUG_BATCH}) for CutMix"

    print("Model:", args.model)

    # =========================
    # LOAD DATA
    # =========================
    train_paths = glob(os.path.join(
        'Datasets/Dataset', args.year, 'Fingerprint',
        'Training', args.scanner, '*', '*', '*'
    ))
    test_paths = glob(os.path.join(
        'Datasets/Dataset', args.year, 'Fingerprint',
        'Testing', args.scanner, '*', '*', '*'
    ))

    X_train, Y_train, y_cat = load_data(train_paths)
    X_test, Y_test, _ = load_data(test_paths)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_cat),
        y=y_cat
    )
    class_weight_dict = dict(enumerate(class_weights))

    # =========================
    # DATASET PIPELINE
    # =========================
    ds_train, ds_test = build_dataset(
        X_train, Y_train,
        X_test, Y_test,
        args
    )

    steps_per_epoch = len(X_train) // args.batch_size
    val_steps = len(X_test) // args.batch_size

    # =========================
    # MODEL
    # =========================
    model = build_model(args.model)

    optimizer = tf.keras.optimizers.Adam(args.lr)
    f1 = tfa.metrics.F1Score(2, average='macro')

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['acc', f1]
    )

    # =========================
    # CHECKPOINT
    # =========================
    out_dir = os.path.join(
        'checkpoint', args.year, args.scanner,
        f'baseline_{args.model}'
    )
    os.makedirs(out_dir, exist_ok=True)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(out_dir, 'model-best-f1.h5'),
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # =========================
    # TRAIN
    # =========================
    model.fit(
        ds_train,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_test,
        validation_steps=val_steps,
        class_weight=class_weight_dict,
        callbacks=[ckpt],
        verbose=1
    )

if __name__ == '__main__':
    main()
