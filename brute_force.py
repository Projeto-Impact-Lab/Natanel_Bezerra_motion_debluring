import tensorflow as tf
import os

from data import Dataset
from models.pixpix import Pix2Pix
from gridsearch import GridSearch
from msg_alert import MsgAlert
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Get the list of physical GPU devices
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for each physical GPU (if available)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


instance = MsgAlert()
def alert(id):
    instance.send_msg(f"Finalizado: {id}")

dataset = Dataset(buffer_size=400, batch_size=1 )


networks = [Pix2Pix]
loss_discriminator = ['disc_loss',]
loss_generator = ['pix2pix_loss','mse_loss','ssim_loss','ssim_mse_loss','acurracy_bce_loss',]
opt_gen_names = ['Adam','AdamX', 'RMSprop', 'SGD']
lrs_gen = np.logspace(-4, -1, 10)
bts_1_gen = [0.2,0.9,0.95,0.85]
bts_2_gen = [0.9999,0.995,0.999,0.99]

opts_disc_name =['Adam',]
lrs_disc = [0.0002,]
bts_1_disc = [0.2,]
bts_2_disc = [0.9999,]


grid = GridSearch(
    networks=networks, 
    dataset=dataset,
    steps=1000,
    loss_discriminator=loss_discriminator,
    loss_generator=loss_generator,
    opt_gen_names=opt_gen_names,
    lrs_gen=lrs_gen,
    bts_1_gen=bts_1_gen,
    bts_2_gen=bts_2_gen,
    opts_disc_name=opts_disc_name,
    lrs_disc=lrs_disc,
    bts_1_disc=bts_1_disc,
    bts_2_disc=bts_2_disc,
    callback=alert )

try:
    grid.run()
except Exception as e:
    print(e)
    instance.send_msg(f"Ocorreu um erros")

instance.send_msg(f"Execução finalizada")