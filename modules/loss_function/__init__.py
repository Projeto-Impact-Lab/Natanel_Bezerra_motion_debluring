import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

class Loss:

    loss_name = None
    loss_function = None
    __mse_loss = tf.keras.losses.MeanSquaredError()
    __kl = tf.keras.losses.KLDivergence()
    __loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    __LAMBDA = None

    def __init__(self, loss_name=None, lambda_val=100) -> None:

        self.loss_name = loss_name
        self.__LAMBDA = lambda_val

        if self.loss_name == 'mse_loss':
            self.loss_function = self.__loss_01
        elif self.loss_name == 'pix2pix_loss':
            self.loss_function = self.__loss_02
        elif self.loss_name == 'disc_loss':
            self.loss_function = self.__discriminator_loss
        elif self.loss_name == 'ssim_loss':
            self.loss_function = self.__loss_03
        elif self.loss_name == 'ssim_mse_loss':
            self.loss_function = self.__loss_04
        elif self.loss_name == 'acurracy_loss':
            self.loss_function = self.__loss_05
        elif self.loss_name == 'ssim_psnr_mse_loss':
            self.loss_function = self.__loss_08
        elif self.loss_name == 'mse_loss_v2':
            self.loss_function = self.__loss_07
        elif self.loss_name == 'psnr_loss':
            self.loss_function = self.__psnr_loss
        elif self.loss_name == 'tv_loss':
            self.loss_function = self.__tv_loss
        elif self.loss_name == 'perceptual_loss':
            self.loss_function = self.__perceptual_loss
        elif self.loss_name == 'edge_loss':
            self.loss_function = self.__edge_loss
        else:
            raise ValueError("Loss invalid")

    @tf.autograph.experimental.do_not_convert
    def __loss_01(self, disc_generated_output=None, gen_output=None, target=None):
        return self.__mse_loss(target, gen_output)
    
    @tf.autograph.experimental.do_not_convert
    def __loss_02(self, disc_generated_output=None, gen_output=None, target=None):
        gan_loss = self.__loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        total_gen_loss = gan_loss + (self.__LAMBDA * l1_loss)
        
        return total_gen_loss
    
    @tf.autograph.experimental.do_not_convert
    def __discriminator_loss(self, disc_real_output=None, disc_generated_output=None):
        real_loss = self.__loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.__loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    @tf.autograph.experimental.do_not_convert
    def __loss_03(self,disc_generated_output=None, gen_output=None, target=None):
        ssim2 = tf.image.ssim(target, gen_output, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
        return 1-ssim2
    

    @tf.autograph.experimental.do_not_convert
    def __loss_04(self,disc_generated_output=None, gen_output=None, target=None):
        ssim2 = tf.image.ssim(target, gen_output, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
        mse_val = self.__mse_loss(target, gen_output)
        return mse_val + (1-ssim2)

    @tf.autograph.experimental.do_not_convert
    def __loss_05(self,disc_generated_output=None, gen_output=None, target=None):

        correct = tf.equal(target, gen_output)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        lamb = 10
        return lamb - accuracy
    
    @tf.autograph.experimental.do_not_convert
    def __loss_06(self, disc_generated_output=None, gen_output=None, target=None):

        correct = tf.equal(target, gen_output)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        kl_resuk = tf.reduce_mean(self.__kl(target, gen_output))
        
        total_gen_loss = (self.__LAMBDA - accuracy) + kl_resuk
        
        return total_gen_loss
    
    @tf.autograph.experimental.do_not_convert
    def __loss_07(self, gen_output=None, target=None):
        return self.__mse_loss(target, gen_output)
    def __loss_08(self, gen_output=None, target=None):
        
        ssim2 = tf.image.ssim(target, gen_output, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
        mse_val = self.__mse_loss(target, gen_output)
        
        psnr_erro = tf.image.psnr(gen_output, target, max_val=1.0)
        
        return (self.__LAMBDA*mse_val) + (self.__LAMBDA*(1-ssim2)) + (50 - psnr_erro)
    
    
    def __psnr_loss(self, gen_output=None, target=None):
        return -tf.reduce_mean(tf.image.psnr(gen_output, target, max_val=1.0))

    def __tv_loss(self, gen_output=None, target=None):
        x_diff = gen_output[:, :, 1:, :] - gen_output[:, :, :-1, :]
        y_diff = gen_output[:, 1:, :, :] - gen_output[:, :-1, :, :]
        return tf.reduce_mean(tf.abs(x_diff)) + tf.reduce_mean(tf.abs(y_diff))

    def __perceptual_loss(self, gen_output=None, target=None):
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        vgg.trainable = False
        model = Model(vgg.input, vgg.get_layer('block5_conv4').output)
        return self.__mse_loss(model(target), model(gen_output))

    def __edge_loss(self, gen_output=None, target=None):
        gen_edges = tf.image.sobel_edges(gen_output)
        target_edges = tf.image.sobel_edges(target)
        return self.__mse_loss(gen_edges, target_edges)