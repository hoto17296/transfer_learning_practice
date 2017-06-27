import tensorflow as tf
import keras
from datetime import datetime

num_classes = 10

# 画像を 224x224 にリサイズするレイヤ
ResizeImageLayer = keras.layers.Lambda(lambda x: tf.image.resize_images(x, (224, 224)), name='resize')

# VGG 用の前処理 (RGB → BGR 変換 + 平均値の差を取る) を行うレイヤ
VGG_MEAN = [103.939, 116.779, 123.68]
def vgg_preprocess(rgb):
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=rgb)
    return tf.concat(axis=3, values=[
        b - VGG_MEAN[0],
        g - VGG_MEAN[1],
        r - VGG_MEAN[2],
    ])
VGGPreprocessLayer = keras.layers.Lambda(vgg_preprocess, name='preprocess')

# 学習済み VGG16 モデルを読み込む
vgg = keras.applications.vgg16.VGG16(weights='imagenet')

# VGG モデルから出力層を削除する
vgg = keras.models.Model(inputs=vgg.input, outputs=vgg.layers[-2].output, name='vgg16')

# VGG モデルの全層の重みを固定する
vgg.trainable = False

# 10クラス分類用の出力層
PredictionsLayer = keras.layers.Dense(num_classes, activation='softmax', name='predictions')

# モデルを構築する
inputs = keras.layers.Input(shape=(32, 32, 3))

x = ResizeImageLayer(inputs)
x = VGGPreprocessLayer(x)
x = vgg(x)
x = PredictionsLayer(x)

model = keras.models.Model(inputs=inputs, outputs=x)

# モデルをコンパイルする
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# CIFAR10 データセットを読み込む
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# TensorBoard 用のログ出力の設定
tensor_board = keras.callbacks.TensorBoard(histogram_freq=1)

# 学習する
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[tensor_board])

# パラメータを保存
model.save_weights('weights/%s.hdf5' % datetime.now().strftime("%Y%m%d%H%M%S"))
