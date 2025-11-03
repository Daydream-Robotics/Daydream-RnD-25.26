import tensorflow as tf
from keras import layers, models, optimizers, regularizers
import tensorboard
import datetime


# CONFIG

l2_reg = regularizers.l2(1e-3) # controls how much weight decay there is


# --------------------------------
# Inverted Residual Linear Bottleneck
# DO NOT CALL
# --------------------------------

def _bottleneck_block(x, expansion_factor, out_channels, stride, name=None):
    in_channels = x.shape[-1]
    expanded_channels = in_channels * expansion_factor

    #1x1 expand by expansion factor
    x_exp = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False, kernel_regularizer=l2_reg, name=f"{name}_expand")(x)
    x_exp = layers.BatchNormalization()(x_exp)
    x_exp = layers.ReLU(6.0)(x_exp)


    #Depthwise singular on expansion
    x_dw = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, depthwise_regularizer=l2_reg, name=f"{name}_depthwise")(x_exp)
    x_dw = layers.BatchNormalization()(x_dw)
    x_dw = layers.ReLU(6.0)(x_dw)

    #1x1 project to out
    x_proj = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, kernel_regularizer=l2_reg, name=f"{name}_projection")(x_dw)
    x_proj = layers.BatchNormalization()(x_proj)

    #residual connection
    if stride==1 and in_channels==out_channels:
        return layers.Add()([x, x_proj])
    else:
        return x_proj
    
def _repeat_bottlenecks(x, expansion_factor, out_channels, repeats, stride, name_prefix):
    for i in range(repeats):
        s = stride if i == 0 else 1
        x = _bottleneck_block(x, expansion_factor, out_channels, s, name=f"{name_prefix}_{i}")
    return x


# --------------------------------
# Model Backbone
# CALL - from model import backbone
# --------------------------------
    
def backbone(input_shape=(320,320,3)):
    inputs = layers.Input(shape=input_shape)

    # x = prediction

    #first conv2d
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2_reg, name="Conv1")(inputs) # 320x320 x 3 in 160x160 x 32 out
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)

    #Bottlenecks
    x = _bottleneck_block(x, 1, 16, 1, name="bneck_1") # 160x160 x 32 in 80x80 x 16 out
    x = _repeat_bottlenecks(x, 6, 24, 2, 2, name_prefix="bneck_2") # 80x80 x 16 in 80x80 x 24 out
    x = _repeat_bottlenecks(x, 6, 32, 3, 1,name_prefix="bneck_3") # 80x80 x 24 in 80x80 x 32 out
    x = _repeat_bottlenecks(x, 6, 64, 3, 1, name_prefix="bneck_4") # 80x80 x 32 in 80x80 x 64 out
    x = _repeat_bottlenecks(x, 6, 96, 3, 2, name_prefix="bneck_5") # 80x80 x 64 in 40x40 x 96 out
    x = _repeat_bottlenecks(x, 6, 128, 2, 1, name_prefix="bneck_6") # 40x40 x 96 in 40x40 x 128 out

    # TAP
    p8 = x

    x = _repeat_bottlenecks(x, 6, 160, 1, 1, name_prefix="bneck_7") # 40x40x128 in 20x20x160 out
    p16 = x

    return models.Model(inputs, [p8, p16], name="backbone")
    