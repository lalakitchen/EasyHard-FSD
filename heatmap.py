import os
import numpy as np
import argparse
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
import matplotlib.pyplot as plt

# =========================
# GPU
# =========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
print("GPUs:", tf.config.list_physical_devices('GPU'))

# =========================
# MODEL (UNCHANGED CORE)
# =========================
def build_model(shape=(256,256,3), NUM_CLASSES=2):
    base = EfficientNetV2S(
        input_shape=shape,
        include_top=False,
        weights='imagenet'
    )
    x = inputs = layers.Input(shape=shape)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    return tf.keras.Model(inputs, outputs)

# =========================
# DATA LOADER (UNCHANGED)
# =========================
def create_bag(datapath):
    imgs, labels = [], []
    for filename in datapath:
        label = filename.split('/')[-2]
        img = cv2.imread(filename)
        img = cv2.resize(img, (256,256))
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)

# =========================
# FIND LAST CONV LAYER (ROBUST)
# =========================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

# =========================
# GRADCAM
# =========================
def compute_gradcam(img, grad_model):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument('--year', default='2015')
    parser.add_argument('--scanner', default='Digital_Persona')
    parser.add_argument('--method', default='baseline',
                        choices=['baseline','hardsample','our'])
    parser.add_argument('--exp_name', default='easyhard')
    parser.add_argument('--max_count', type=int, default=50)
    args = parser.parse_args()
    print(args)

    # =========================
    # LOAD DATA
    # =========================
    test_path = glob(os.path.join(
        'Datasets', 'Dataset', args.year,
        'Fingerprint', 'Testing',
        args.scanner, '*', '*', '*'
    ))
    X_test, y_test = create_bag(test_path)

    unique_labels = np.unique(y_test)
    for lab in unique_labels:
        os.makedirs(f'Visualizations_hard/{lab}/original', exist_ok=True)
        os.makedirs(f'Visualizations_hard/{lab}/heatmap', exist_ok=True)
        os.makedirs(f'Visualizations_hard/{lab}/ori_heatmap', exist_ok=True)

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

    print("Loading:", ckpt)
    model.load_weights(ckpt)

    last_conv = find_last_conv_layer(model)
    grad_model = tf.keras.Model(
        [model.input],
        [model.get_layer(last_conv).output, model.output]
    )

    # =========================
    # VISUALIZATION LOOP
    # =========================
    for i, (img, lab) in enumerate(zip(X_test, y_test)):
        if i >= args.max_count:
            break

        img_exp = tf.cast(img[None, ...], tf.float32)
        heatmap = compute_gradcam(img_exp, grad_model)
        heatmap = cv2.resize(heatmap, (256,256))

        # SAVE
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f'Visualizations_hard/{lab}/original/{i}.jpg', bbox_inches='tight')
        plt.close()

        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')
        plt.savefig(f'Visualizations_hard/{lab}/heatmap/{i}.jpg', bbox_inches='tight')
        plt.close()

        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.savefig(f'Visualizations_hard/{lab}/ori_heatmap/{i}.jpg', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
