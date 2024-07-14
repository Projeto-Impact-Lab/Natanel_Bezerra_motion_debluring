from config import Config
from optimizers import Optimizers
from loss_function import Loss
from saved_results import SavedResults
import tensorflow as tf


class GridSearch(Config):

    networks = None
    dataset = None
    steps = None
    loss_discriminator = None
    loss_generator = None
    opt_gen_names = None
    lrs_gen = None
    bts_1_gen = None
    bts_2_gen = None
    opts_disc_name = None
    lrs_disc = None
    bts_1_disc = None
    bts_2_disc = None
    total = None

    callback = None


    mse_loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, networks=None, 
                 dataset=None, 
                 steps=50000, 
                 loss_discriminator=None, 
                 loss_generator=None,
                    opt_gen_names = None,
                    lrs_gen = None,
                    bts_1_gen = None,
                    bts_2_gen = None,
                    opts_disc_name = None,
                    lrs_disc = None,
                    bts_1_disc = None,
                    bts_2_disc = None,
                    callback = None
                 ) -> None:
        super().__init__()

        self.networks = networks 
        self.dataset = dataset
        self.steps = steps
        self.loss_discriminator = loss_discriminator
        self.loss_generator = loss_generator
        self.opt_gen_names = opt_gen_names
        self.lrs_gen = lrs_gen
        self.bts_1_gen = bts_1_gen
        self.bts_2_gen = bts_2_gen
        self.opts_disc_name = opts_disc_name
        self.lrs_disc = lrs_disc
        self.bts_1_disc = bts_1_disc
        self.bts_2_disc = bts_2_disc
        self.callback = callback

        self.total = len(self.networks) * len(self.loss_discriminator) * len(self.loss_generator) * len( self.opt_gen_names)
        self.total = self.total * len(self.lrs_gen) * len(self.bts_1_gen) * len(self.bts_2_gen) * len(self.opts_disc_name)
        self.total = self.total * len(self.lrs_disc) * len(self.bts_1_disc) * len(self.bts_2_disc )

    def score(self, y_true, y_pred):

        correct = tf.equal(y_true, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
    
    def image_evaluation(self, model, test_input, tar):
    
        prediction = model(test_input, training=True)

        score_model = self.score(tar, prediction)
        score_mse = self.mse_loss(tar, prediction)
        score_ssim = tf.image.ssim(tar, prediction, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
        score_psnr = tf.image.psnr(tar, prediction, max_val=1.0)

        return score_model,score_mse, score_ssim[0], score_psnr[0]
    
    def run(self,):
        cont = 1
        for net in self.networks:
            for gen in self.loss_generator:
                for disc_name in self.loss_discriminator:
                    for opt_gen_name in self.opt_gen_names:
                        for lr_gen in self.lrs_gen:
                            for bt_1_gen in self.bts_1_gen:
                                for bt_2_gen in self.bts_2_gen:
                                    for opt_disc_name in self.opts_disc_name:
                                        for lr_disc in self.lrs_disc:
                                            for bt_1_disc in self.bts_1_disc:
                                                for bt_2_disc in self.bts_2_disc:
                                                    self.gen_exec(net, gen, disc_name, 
                                                    opt_gen_name,lr_gen,bt_1_gen, bt_2_gen,
                                                    opt_disc_name,lr_disc,bt_1_disc, bt_2_disc)

                                                    if self.callback != None:
                                                        self.callback(f'{cont}/{self.total}')
                                                        cont+=1



                

    def gen_exec(self,net, gen_name, disc_name, 
                 opt_gen_name,lr_gen,bt_1_gen, bt_2_gen,
                 opt_disc_name,lr_disc,bt_1_disc, bt_2_disc):
        generator_loss = Loss(gen_name)
        discriminator_loss = Loss(disc_name)

        generator_optimizer = Optimizers(name_opt=opt_gen_name,
                                         learning_rate=lr_gen, beta_1=bt_1_gen, beta_2=bt_2_gen).optimizer
        
        discriminator_optimizer = Optimizers(name_opt=opt_disc_name,learning_rate=lr_disc, beta_1=bt_1_disc, beta_2=bt_2_disc).optimizer

        pix = net()

        pix.compile(
        otimazer_generator=generator_optimizer,
        otimazer_discriminator=discriminator_optimizer, 
        loss_generator=generator_loss.loss_function, 
        loss_discriminator=discriminator_loss.loss_function
        )

        pix.fit(self.dataset.train, self.dataset.test, steps=self.steps, checkpoint_save=False)

        # Run the trained model on a few examples from the test set

        inp, tar = self.dataset.upload_specific_image("1_M03-1326_000172")
        acurracy, mse, ssim, psnr = self.image_evaluation(pix.generator, inp, tar)
        generator_optimizer_settigs = generator_optimizer.get_config()

        SavedResults(id=1, mse=mse.numpy(), ssim=ssim.numpy(), parameters=f'Loss: {gen_name}, Optimazer:{generator_optimizer_settigs}')
        
        del pix
