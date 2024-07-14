import os
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.data import Dataset
from modules.models.deblur import Deblur
from modules.loss_function import Loss
from modules.optimizers import Optimizers

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Get the list of physical GPU devices
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for each physical GPU (if available)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
        

def main(name="1",steps=100, loss_name='mse_loss', optimazer_name="Adam", lr=0.001, bt_1=0.95, bt_2=0.999):

    dataset = Dataset(buffer_size=400, batch_size=1 )
    test_data = dataset.test_imgs()

    generator_loss = Loss(loss_name).loss_function

    generator_optimizer = Optimizers(name_opt=optimazer_name,
                                    learning_rate=lr,
                                    beta_1=bt_1, beta_2=bt_2
                                    ).optimizer
    net = Deblur()
    net.compile(
        otimazer_generator=generator_optimizer,
        loss_generator=generator_loss
        )


    net.fit(dataset.train, test_data, steps=steps, checkpoint_save=False)

    if net.pontuations != None:
        pontuation = (1-net.pontuations[0]) + net.pontuations[1]
        np.save(f"./outputs/{name}", pontuation)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('steps', type=int, help='an integer for total steps')
    parser.add_argument('loss_name', type=str, help='a string value')
    parser.add_argument('optimazer_name', type=str, help='a string value')
    parser.add_argument('lr', type=float, help='a float value')
    parser.add_argument('bt_1', type=float, help='a float value')
    parser.add_argument('bt_2', type=float, help='a float value')
    parser.add_argument('name', type=str, help='a string value')

    args = parser.parse_args()

    main(steps=args.steps, loss_name=args.loss_name, optimazer_name=args.optimazer_name, 
         lr=args.lr, bt_1=args.bt_1, bt_2=args.bt_2, name=args.name)