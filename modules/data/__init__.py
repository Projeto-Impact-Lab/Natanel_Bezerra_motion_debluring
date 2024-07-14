
import tensorflow as tf
import gdown
import subprocess

from config import Config
import os
import numpy as np

import random

class Dataset(Config):

    train = None
    test = None
    batch_size = None
    shape = None
    buffer_size = None
    format = None
    

    def __init__(self, batch_size=1, buffer_size=None , format='png', shape=(256,256,3)) -> None:
        
        super().__init__()
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.format =format
        self.shape = shape

        if self.train == None:

            if not self.base_already_download():
                print("download...")
                self.download()
            
            if buffer_size == None:
                self.buffer_size = len( list(tf.data.Dataset.list_files(f'{self.downloads}/{self.dataset_name}/train/*.{self.format}' )) )
                
            self.train= tf.data.Dataset.list_files(f'{self.downloads}/{self.dataset_name}/train/*.{self.format}').map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE).shuffle(self.buffer_size).batch(self.batch_size)            
            self.test = tf.data.Dataset.list_files(f'{self.downloads}/{self.dataset_name}/test/*.{self.format}').map(self.load_image_test).shuffle(self.buffer_size).batch(self.batch_size)    
            self.test = self.test.take(self.buffer_size)
            print("base loaded.")

    def download(self,):

        gdown.download(url=self.url_download, 
                    output=f'{self.downloads}/{self.dataset_file}', 
                    quiet=False, fuzzy=True)

        if self.dataset_extension == ".tar.gz":
            tar_command = ["tar", "-xvf", f'{self.downloads}/{self.dataset_file}', '-C', f'{self.downloads}']
            subprocess.run(tar_command, check=True)
        elif self.dataset_extension == ".zip":
            tar_command = ["unzip", f'{self.downloads}/{self.dataset_file}', '-d', f'{self.downloads}']
            subprocess.run(tar_command, check=True)
        else:
            raise ValueError("Not implementation for this extension")


    def base_already_download(self):
        file_path = os.path.join(self.downloads, f'{self.dataset_name}{self.dataset_extension}')
        return os.path.isfile(file_path)
    
    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_png(image)

        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)

        w,h,c = self.shape

        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, w, h, c])

        return cropped_image[0], cropped_image[1]

    # Normalizing the images to [-1, 1]
    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        # Resizing to 286x286
        x,y = random.choice( [ (286,286),(512,512), (286,286), (1024,1024), (286,286) ] )
        input_image, real_image = self.resize(input_image, real_image, x,y)

        # Random cropping back to 256x256
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image,
                                        256, 256)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
    
    def upload_specific_image(self,name):
        img_origin, img_transf = self.load(f'{self.downloads}/{self.dataset_name}/test/{name}.{self.format}')
        w,h,c = self.shape
        img_origin, img_transf = self.resize(img_origin, img_transf, w, h)
        img_origin, img_transf = self.normalize(img_origin, img_transf)

        img_origin = img_origin[tf.newaxis,...]
        img_transf = img_transf[tf.newaxis,...]

        return img_origin, img_transf
        
    def test_imgs(self,qnt=100, force_save=False):

        file_path = os.path.join(self.downloads, f'test.npy')

        if (not os.path.isfile(file_path)) or force_save:
            test_data = [(inp_batch, tar_batch) for (inp_batch, tar_batch) in self.test.take(qnt)]
            test_data = np.array(test_data)

            with open(f"{self.downloads}/test.npy","wb") as f:
                np.save(f, test_data)

            return test_data
        else:

            test_data = np.load(f"{self.downloads}/test.npy")
            return test_data
