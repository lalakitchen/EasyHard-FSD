import os
import numpy as np
import argparse
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.utils import class_weight
from tensorflow.keras.losses import categorical_crossentropy, KLDivergence

from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetV2S
)

# =========================
# IMPORT IMAGE GENERATOR
# =========================
from utils.ImageGenerator import build_dataset, AUG_BATCH

# =========================
# GPU
# =========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
print("GPUs:", tf.config.list_physical_devices('GPU'))

# =========================
# LABEL
# =========================
def class_map(cat):
    y = np.zeros(2, dtype=np.float32)
    y[0 if cat == 'Live' else 1] = 1
    return y

# =========================
# DATA (NUMPY LOADER)
# =========================
def load_data(paths):
    X, Y, cats = [], [], []
    for p in paths:
        try:
            cat = p.split('/')[-3]
            img = cv2.imread(p)
            img = cv2.resize(img, (256,256))
            X.append(img)
            Y.append(class_map(cat))
            cats.append(cat)
        except:
            print("Error:", p)
    return np.array(X), np.array(Y), np.array(cats)

# =========================
# MODEL FACTORY
# =========================
def build_model(name, input_shape=(256,256,3), num_classes=2):

    if name == 'resnet':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        x = layers.Input(shape=input_shape)
        y = base(x)
        y = layers.GlobalAveragePooling2D()(y)

    elif name == 'effnet':
        base = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=input_shape)
        x = layers.Input(shape=input_shape)
        y = base(x)
        y = layers.GlobalAveragePooling2D()(y)

    elif name == 'vit':
        x = layers.Input(shape=input_shape)
        z = layers.Rescaling(1./255)(x)

        patch = 16
        n_patches = (256 // patch) ** 2
        proj = 64

        z = tf.image.extract_patches(
            images=z,
            sizes=[1, patch, patch, 1],
            strides=[1, patch, patch, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        z = tf.reshape(z, [-1, n_patches, patch*patch*3])
        z = layers.Dense(proj)(z)

        for _ in range(4):
            attn = layers.MultiHeadAttention(4, proj)(z, z)
            z = layers.LayerNormalization()(z + attn)
            mlp = layers.Dense(proj*2, activation='gelu')(z)
            mlp = layers.Dense(proj)(mlp)
            z = layers.LayerNormalization()(z + mlp)

        y = layers.GlobalAveragePooling1D()(z)

    else:
        raise ValueError("model ∈ {resnet, effnet, vit}")

    y = layers.Dropout(0.2)(y)
    out = layers.Dense(num_classes, activation='softmax')(y)
    return tf.keras.Model(x, out)

# =========================
# EMA
# =========================
def ema_update(teacher, student, alpha):
    for tw, sw in zip(teacher.weights, student.weights):
        tw.assign(alpha * tw + (1 - alpha) * sw)

# =========================
# HARD MINING (NUMPY SPACE)
# =========================
def hard_mining(X, Y, teacher, eta):
    pred = teacher.predict(X, verbose=0)
    loss = categorical_crossentropy(Y, pred).numpy()
    N = int(len(X) * eta)
    idx = np.argsort(loss)[-N:]
    return X[idx], Y[idx]

# =========================
# KD LOSS
# =========================
def kd_loss(y_true, y_pred, teacher_pred, T=4.0, alpha=0.7):
    ce = categorical_crossentropy(y_true, y_pred)
    kl = KLDivergence()(
        tf.nn.softmax(teacher_pred / T),
        tf.nn.softmax(y_pred / T)
    )
    return alpha * ce + (1 - alpha) * kl * (T**2)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet','effnet','vit'])
    parser.add_argument('--year', default='2015')
    parser.add_argument('--scanner', default='CrossMatch')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--Eg', type=int, default=3)
    parser.add_argument('--Et', type=int, default=5)
    parser.add_argument('--Es', type=int, default=20)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--ema', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--exp', required=True)
    args = parser.parse_args()

    # =========================
    # SAFETY
    # =========================
    assert args.batch == AUG_BATCH, \
        f"--batch must equal AUG_BATCH ({AUG_BATCH}) for CutMix"

    # =========================
    # LOAD DATA
    # =========================
    train_paths = glob(os.path.join(
        'Datasets/Dataset', args.year, 'Fingerprint',
        'Training', args.scanner, '*', '*', '*'
    ))
    X, Y, cats = load_data(train_paths)

    cw = class_weight.compute_class_weight(
        'balanced', classes=np.unique(cats), y=cats
    )
    cw = dict(enumerate(cw))

    # =========================
    # MODELS
    # =========================
    teacher = build_model(args.model)
    student = build_model(args.model)

    f1 = tfa.metrics.F1Score(2, average='macro')

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='categorical_crossentropy',
        metrics=['acc', f1]
    )

    student.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='categorical_crossentropy',
        metrics=['acc', f1]
    )

    # =========================
    # TRAINING LOOP
    # =========================
    for eg in range(args.Eg):
        print(f"\n=== Global Iter {eg+1} ===")

        # -------- Teacher Training (with augmentation) --------
        ds_teacher, _ = build_dataset(X, Y, X, Y, args)
        steps = len(X) // args.batch

        teacher.fit(
            ds_teacher,
            steps_per_epoch=steps,
            epochs=args.Et,
            class_weight=cw,
            verbose=1
        )

        # -------- Hard Mining (clean images) --------
        Xh, Yh = hard_mining(X, Y, teacher, args.eta)

        # -------- Student KD Training (augmented batches) --------
        teacher_logits = teacher.predict(Xh, verbose=0)

        ds_student, _ = build_dataset(Xh, Yh, Xh, Yh, args)
        ds_student = ds_student.take(len(Xh) // args.batch)

        for _ in range(args.Es):
            for xb, yb in ds_student:
                with tf.GradientTape() as tape:
                    sp = student(xb, training=True)
                    tp = teacher_logits[:len(xb)]
                    loss = kd_loss(yb, sp, tp)
                grads = tape.gradient(loss, student.trainable_weights)
                student.optimizer.apply_gradients(
                    zip(grads, student.trainable_weights)
                )

        # -------- EMA Update --------
        ema_update(teacher, student, args.ema)

        # -------- SAVE --------
        out = os.path.join(
            'checkpoint', args.year, args.scanner,
            f'{args.exp}_{args.model}_eg{eg}'
        )
        os.makedirs(out, exist_ok=True)
        teacher.save(os.path.join(out, 'teacher.h5'))
        student.save(os.path.join(out, 'student.h5'))

if __name__ == '__main__':
    main()
