import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from IPython import display

class Pix2Pix:

  name_arq = None
  generator = None
  discriminator = None
  loss_generator = None
  loss_discriminator = None
  otimazer_discriminator = None
  otimazer_generator = None
  checkpoint_dir  = None
  checkpoint_prefix = None
  checkpoint = None

  def __init__(self,):
    self.name_arq = "Pix2Pix"

    if self.generator == None:
      self.create_generator()

    if self.discriminator == None:
      self.create_disciminator()

  def __str__(self):
        return f"Pix2Pix(name_arq={self.name_arq})"

  def _downsampling(self, filters, size, apply_batchNormalize=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add( tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))


    if apply_batchNormalize:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

  def _upsampling(self, filters, size, apply_dropout=False):

    initialize = tf.random_normal_initializer(0., 0.02)

    up = keras.Sequential()

    up.add(
        keras.layers.Conv2DTranspose(filters, size, strides=2 ,padding='same', kernel_initializer=initialize, use_bias=False)
    )

    up.add(
        keras.layers.BatchNormalization()
    )

    if apply_dropout:
      up.add(
          keras.layers.Dropout(0.5)
      )

    up.add(
        keras.layers.LeakyReLU()
    )

    return up

  def create_generator(self,):

    inputs = keras.layers.Input(shape=[256,256,3])

    down_stack = [
    self._downsampling(64, 4, apply_batchNormalize=False),  # (batch_size, 128, 128, 64)
    self._downsampling(128, 4),  # (batch_size, 64, 64, 128)
    self._downsampling(256, 4),  # (batch_size, 32, 32, 256)
    self._downsampling(512, 4),  # (batch_size, 16, 16, 512)
    self._downsampling(512, 4),  # (batch_size, 8, 8, 512)
    self._downsampling(512, 4),  # (batch_size, 4, 4, 512)
    self._downsampling(512, 4),  # (batch_size, 2, 2, 512)
    self._downsampling(512, 4),  # (batch_size, 1, 1, 512)
  ]

    uping_stack = [
    self._upsampling(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    self._upsampling(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    self._upsampling(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    self._upsampling(512, 4),  # (batch_size, 16, 16, 1024)
    self._upsampling(256, 4),  # (batch_size, 32, 32, 512)
    self._upsampling(128, 4),  # (batch_size, 64, 64, 256)
    self._upsampling(64, 4),  # (batch_size, 128, 128, 128)
  ]

    initializer = tf.random_normal_initializer(0.,0.02)

    last = keras.layers.Conv2DTranspose(3,4, strides=2, padding='same', kernel_initializer=initializer, activation='elu')


    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(uping_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)


    self.generator = keras.Model(inputs=inputs, outputs=x)

  def create_disciminator(self):

    initializer = tf.random_normal_initializer(0.,0.02)

    ipt = keras.layers.Input(shape=[256,256,3], name='input_image')
    tar = keras.layers.Input(shape=[256,256,3], name='target_image')

    conc = keras.layers.concatenate([ipt,tar])

    down1 = self._downsampling( 64, 4, False)(conc)
    down2 = self._downsampling(128, 4)(down1)
    down3 = self._downsampling(256, 4)(down2)

    zero_pad1 = keras.layers.ZeroPadding2D()(down3)

    conv = keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

    ins_batch = keras.layers.BatchNormalization()(conv)
    leak = keras.layers.LeakyReLU()(ins_batch)
    zero_pad2 = keras.layers.ZeroPadding2D()(leak)
    last = keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)


    self.discriminator = tf.keras.Model(inputs=[ipt, tar], outputs=last)

  def compile(self, loss_generator=None, loss_discriminator=None, otimazer_generator=None, otimazer_discriminator=None, checkpoint_dir="./"):

    self.loss_generator = loss_generator
    self.loss_discriminator = loss_discriminator
    self.otimazer_generator = otimazer_generator
    self.otimazer_discriminator = otimazer_discriminator

    self.checkpoint_dir     = checkpoint_dir
    self.checkpoint_prefix  = os.path.join(self.checkpoint_dir, "ckpt")
    self.checkpoint         = tf.train.Checkpoint(generator_optimizer=self.otimazer_generator,
                              discriminator_optimizer=self.otimazer_discriminator ,
                              generator=self.generator,
                              discriminator=self.discriminator)
    self.restoration()

  @tf.function
  def step_fit(self, input_image, target, step, log_saved=None):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      gen_output = self.generator(input_image, training=True)

      disc_output_real = self.discriminator([input_image, target], training=True)
      disc_output_fake = self.discriminator([gen_output, target], training=True)

      gen_loss_total = self.loss_generator(disc_output_fake, gen_output, target)
      disc_loss_total = self.loss_discriminator(disc_output_real, disc_output_fake)

    gradients_generator = gen_tape.gradient(gen_loss_total, self.generator.trainable_variables)
    gradients_discriminator = disc_tape.gradient(disc_loss_total, self.discriminator.trainable_variables)

    self.otimazer_generator.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
    self.otimazer_discriminator.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))

    if log_saved != None:
      log_saved(gen_loss_total=gen_loss_total,
                disc_loss_total=disc_loss_total,
                step=step)

  def fit(self, train_ds, test_ds, steps, view_callback=None, log_saved=None, checkpoint_save=True):

    if view_callback != None:
      example_input, example_target = next(iter(test_ds.take(1)))

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step) % 1000 == 0:

        display.clear_output(wait=True)

        if view_callback != None:
          view_callback(self.generator, example_input, example_target)

        print(f"Step: {step//1000}k")

      self.step_fit(input_image, target, step, log_saved)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)

      # Save (checkpoint) the model every 5k steps
      if checkpoint_save and ((step + 1) % 5000 == 0):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

  def restoration(self,):
    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
  
  def score(self, y_true, y_pred):
    """Computes the accuracy of the model.

    Args:
      y_true: The ground truth labels.
      y_pred: The predicted labels.

    Returns:
      The accuracy of the model.
    """

    correct = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy