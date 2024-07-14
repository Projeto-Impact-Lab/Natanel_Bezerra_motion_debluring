import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from IPython import display

class Deblur:

  name_arq = None
  generator = None
  loss_generator = None
  otimazer_generator = None
  checkpoint_dir  = None
  checkpoint_prefix = None
  checkpoint = None
  pontuations = None

  def __init__(self):
    self.name_arq = "deblur"
    

    if self.generator == None:
      self.create_generator()

  def __str__(self):
        return f"Deblur(name_arq={self.name_arq})"

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


  def compile(self, loss_generator=None, otimazer_generator=None, checkpoint_dir="./", max_to_keep=1):

    self.loss_generator = loss_generator
    self.otimazer_generator = otimazer_generator

    self.checkpoint_dir     = checkpoint_dir
    self.checkpoint_prefix  = os.path.join(self.checkpoint_dir, "ckpt")
    self.checkpoint         = tf.train.Checkpoint(generator=self.generator,generator_optimizer=self.otimazer_generator)
    self.max_to_keep = max_to_keep

    self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=self.max_to_keep)

    self.restoration()

  @tf.function
  def step_fit(self, input_image, target, step, log_saved=None):

    with tf.GradientTape() as gen_tape:

      gen_output = self.generator(input_image, training=True)
      gen_loss_total = self.loss_generator(gen_output=gen_output,  target=target)

    gradients_generator = gen_tape.gradient(gen_loss_total, self.generator.trainable_variables)

    self.otimazer_generator.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))

    if log_saved != None:
      log_saved(gen_loss_total=gen_loss_total,
                step=step)
    
    return tf.identity(gen_loss_total)

  def fit(self, train_ds, test_ds, steps, view_callback=None, log_saved=None, checkpoint_save=True,
           error_save=None, send_update=None, time_wait=5000):

    if view_callback != None:
      example_input, example_target = next(iter(test_ds.take(1)))

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step+1) % 1000 == 0:

        display.clear_output(wait=True)

        if view_callback != None:
          view_callback(self.generator, example_input, example_target, (step+1)//1000)

        print(f"Step: {step//1000}k")

      error = self.step_fit(input_image, target, step, log_saved)

      if error_save != None and (step+1)%5000==0:
        error_save(step/5000, error.numpy())
    
      if send_update != None and (step+1)%time_wait==0:
        send_update(step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)

      # Save (checkpoint) the model every 5k steps
      if checkpoint_save and ((step + 1) % 5000 == 0):
        self.checkpoint_manager.save()

    display.clear_output(wait=True)
    print("Avaliando modelo")
    ssim_resul, mse_resul, psnr_total = self.__evaluation(test_ds)
    display.clear_output(wait=True)
    
    self.pontuations = [tf.identity(ssim_resul).numpy(), tf.identity(mse_resul).numpy(),tf.identity(psnr_total).numpy()]
    
    if checkpoint_save:
        self.checkpoint_manager.save()

  def restoration(self,):
    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

  def __evaluation(self, test_dataset):
    ssim_total = []
    mse_total = []
    psnr_total = []
    cont = 0
    for inp_batch, tar_batch in test_dataset:
      if (cont+1) % 10 == 0:
        print('.', end='', flush=True)
      cont += 1
      predict = self.generator(inp_batch, training=True)
      ssim_total.append( tf.image.ssim(predict, tar_batch, max_val=1.0) )
      mse_total.append(  tf.losses.mean_squared_error(tar_batch,predict) )
      psnr_total.append( tf.image.psnr(predict, tar_batch, max_val=1.0) )

    return tf.reduce_mean(ssim_total),tf.reduce_mean(mse_total),tf.reduce_mean(psnr_total)