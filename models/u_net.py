import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Add,
    Concatenate
)

import os
import sys
MODULE_PATH = os.path.abspath(f"{os.path.split(__file__)[0]}/..")
if sys.path[0] != MODULE_PATH: sys.path.insert(0, MODULE_PATH)
    
from models.t2t import local_attention_2d, split_heads_2d, combine_heads_2d


def conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dilation_rate=1, dropout_rate=0.4):
    """Convolutional encoder block of U-net, encoder block does not downsample the input feature"""

    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=strides, dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=(1, 1), dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    if strides != (1, 1):
        skip = Conv2D(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def transpose_conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dropout_rate=0.4):
    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=(1, 1), padding="same")(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2DTranspose(channel, kernel_size, strides=strides, padding="same")(input_tensor)

    if strides != (1, 1):
        skip = Conv2DTranspose(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def semantic_segmentation(feature_num=352, timesteps=256, multi_grid_layer_n=1, multi_grid_n=5, ch_num=1, out_class=2, dropout=0.4):
    """Improved U-net model with Atrous Spatial Pyramid Pooling (ASPP) block"""

    input_score = Input(shape=(timesteps, feature_num, ch_num), name="input_score_48")
    en = Conv2D(2**7, (7, 7), strides=(1, 1), padding="same")(input_score)

    en_l1 = conv_block(en, 2**7, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2**7, (3, 3), strides=(1, 1))

    en_l2 = conv_block(en_l1, 2**7, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2**7, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2**7, (3, 3), strides=(1, 1))

    en_l3 = conv_block(en_l2, 2**7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))

    en_l4 = conv_block(en_l3, 2**8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))

    feature = en_l4
    for _ in range(multi_grid_layer_n):
        feature = BatchNormalization()(Activation("relu")(feature))
        feature = Dropout(dropout)(feature)
        m = BatchNormalization()(Conv2D(2**9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
        multi_grid = m
        for ii in range(multi_grid_n):
            m = BatchNormalization()(
                Conv2D(2**9, (3, 3), strides=(1, 1), dilation_rate=2**ii, padding="same", activation="relu")(feature)
            )
            multi_grid = Concatenate()([multi_grid, m])
        multi_grid = Dropout(dropout)(multi_grid)
        feature = Conv2D(2**9, (1, 1), strides=(1, 1), padding="same")(multi_grid)

    feature = BatchNormalization()(Activation("relu")(feature))

    feature = Conv2D(2**8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = Add()([feature, en_l4])
    de_l1 = transpose_conv_block(feature, 2**7, (3, 3), strides=(2, 2))

    skip = de_l1
    de_l1 = BatchNormalization()(Activation("relu")(de_l1))
    de_l1 = Concatenate()([de_l1, BatchNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(dropout)(de_l1)
    de_l1 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = Add()([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2**7, (3, 3), strides=(2, 2))

    skip = de_l2
    de_l2 = BatchNormalization()(Activation("relu")(de_l2))
    de_l2 = Concatenate()([de_l2, BatchNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(dropout)(de_l2)
    de_l2 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = Add()([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2**7, (3, 3), strides=(2, 2))

    skip = de_l3
    de_l3 = BatchNormalization()(Activation("relu")(de_l3))
    de_l3 = Concatenate()([de_l3, BatchNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(dropout)(de_l3)
    de_l3 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = Add()([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2**7, (3, 3), strides=(2, 2))

    de_l4 = BatchNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(dropout)(de_l4)
    out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", name="prediction")(de_l4)

    return Model(inputs=input_score, outputs=out)