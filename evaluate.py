import os
import numpy as np
import argparse
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S

# =========================
# IMPORT IMAGE GENERATOR
# =========================
from utils.ImageGenerator import val_process

# =========================
# GPU SETUP
# =========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# =========================
# LABEL
# =========================
def class_map(category):
    label = np.zeros(2, dtype=np.float32)
    label[0 if category == 'Live' else 1] = 1
    return label

# =========================
# DATA LOADER (NUMPY)
# =========================
def create_bag(datapath):
    imgs, one_hot_labels, labels = [], [], []
    for filename in datapath:
        try:
            label = filename.split('/')[-3]
            img = cv2.imread(filename)
            img = cv2.resize(img, (256,256))
            imgs.append(img)
            one_hot_labels.append(class_map(label))
            labels.append(label)
        except:
            print('image not found:', filename)
    return np.array(imgs), np.array(one_hot_labels), np.array(labels)

# =========================
# MODEL (IDENTICAL TO TRAIN)
# =========================
def build_model(shape=(256,256,3), NUM_CLASSES=2):
    base = EfficientNetV2S(
        input_shape=shape,
        include_top=False,
        weights='imagenet'
    )
    inputs = layers.Input(shape=shape)
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    return tf.keras.Model(inputs, outputs)

# =========================
# ACE METRIC (UNCHANGED)
# =========================
def compute_ACE(model, X_live, y_live, X_fake, y_fake):

    y_live_preds = model.predict(X_live, verbose=0).argmax(axis=1)
    y_live = y_live.argmax(axis=1)

    y_fake_preds = model.predict(X_fake, verbose=0).argmax(axis=1)
    y_fake = y_fake.argmax(axis=1)

    far = (y_live_preds != y_live).sum() / len(y_live)
    frr = (y_fake_preds != y_fake).sum() / len(y_fake)
    ace = (far + frr) / 2

    print(f"FAR : {far:.4f}")
    print(f"FRR : {frr:.4f}")
    print(f"ACE : {ace:.4f}")

# =========================
# TF.DATA HELPER (EVAL ONLY)
# =========================
def build_eval_dataset(X, Y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(val_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Fingerprint PAD')
    parser.add_argument('--year', default='2015')
    parser.add_argument('--scanner', default='Digital_Persona')
    parser.add_argument('--method', default='baseline',
                        choices=['baseline', 'hardsample', 'our'])
    parser.add_argument('--model', default='effnet',
                        choices=['effnet'])
    parser.add_argument('--exp_name', default='baseline')
    args = parser.parse_args()
    print(args)

    # =========================
    # LOAD TEST DATA
    # =========================
    live_test_path = glob(os.path.join(
        'Datasets', 'Dataset', args.year,
        'Fingerprint', 'Testing',
        args.scanner, 'Live', '*', '*'
    ))
    fake_test_path = glob(os.path.join(
        'Datasets', 'Dataset', args.year,
        'Fingerprint', 'Testing',
        args.scanner, 'Fake', '*', '*'
    ))

    live_X, live_Y, _ = create_bag(live_test_path)
    fake_X, fake_Y, _ = create_bag(fake_test_path)

    # =========================
    # LOAD MODEL
    # =========================
    model = build_model()

    if args.method == 'baseline':
        ckpt = f'checkpoint/{args.year}/{args.scanner}/baseline/model-best-f1.h5'
    elif args.method == 'hardsample':
        ckpt = f'checkpoint/{args.year}/{args.scanner}/hardsample/model-best-f1.h5'
    else:
        ckpt = f'checkpoint/{args.year}/{args.scanner}/{args.exp_name}/teacher.h5'

    print("Loading checkpoint:", ckpt)
    model.load_weights(ckpt)

    # =========================
    # EVALUATE
    # =========================
    print('[Testing Evaluation]')
    compute_ACE(
        model,
        live_X, live_Y,
        fake_X, fake_Y
    )

if __name__ == '__main__':
    main()
