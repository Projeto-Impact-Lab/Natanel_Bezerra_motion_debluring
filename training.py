import numpy as np
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
from modules.data import Dataset
from modules.models.deblur import Deblur
from modules.loss_function import Loss
from modules.optimizers import Optimizers
from modules.plots_easy import PlotsEasy
from modules.msg_alert import MsgAlert


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Get the list of physical GPU devices
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth for each physical GPU (if available)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


alert = MsgAlert()
dataset = Dataset(buffer_size=400, batch_size=1 )

ssim_vals = PlotsEasy(xlabel='Step', ylabel='SSIM', title='Step X SSIM')
errors = PlotsEasy(xlabel='Step', ylabel='loss_error', title='Step X Loss_error')
mse = PlotsEasy(xlabel='Step', ylabel='Mse', title='Step X Mse')

def view_images(model, test_input, tar, epoch=1):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    ssim_rs = tf.image.ssim(tar, prediction, max_val=1.0)
    ssim_vals.append(epoch,ssim_rs.numpy())
    
    mse_error = tf.reduce_mean(tf.keras.losses.mean_squared_error(tar, prediction))
    mse.append(epoch,mse_error.numpy())

    ssim_vals.save('./imgs_for_send/ssim_results.png')
    errors.save('./imgs_for_send/loss_results.png')
    mse.save('./imgs_for_send/mse_results.png')
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        
        # Convert to integers in the range [0, 255]
        clipped_image = np.clip((display_list[i] + 1) * 0.5, 0, 1)
        int_image = (clipped_image * 255).astype(np.uint8)
        plt.imshow(int_image)
        
        plt.axis('off')

    plt.suptitle(f"Result - Epoch {epoch}", fontsize=16)
    plt.savefig('./imgs_for_send/temp_result.png')
    plt.clf()
    plt.close()

   

def save_error(epoc, error):
    errors.append(epoc, error)

def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_paths.append(os.path.join(directory, filename))
    return image_paths

def send_update(epoch):
    alert.send_msg(msg=f"Hey chefe, aqui suas atualizações. Estamos na epoca: {epoch}")

    for path in get_image_paths('./imgs_for_send'):
        name = path.split('/')[-1]
        alert.send_msg(msg=name, image_path=path)



def main(steps=100, loss_name='mse_loss', optimazer_name="Adam", lr=0.001, bt_1=0.95, bt_2=0.999):
    try:
        loss = Loss(loss_name).loss_function
        optimazer = Optimizers(name_opt=optimazer_name, 
                               learning_rate=lr, beta_1=bt_1, beta_2=bt_2).optimizer
        net = Deblur()
        checkpoint_dir = './training_checkpoints'
        net.compile(
                otimazer_generator=optimazer,
                loss_generator=loss,
                checkpoint_dir=checkpoint_dir
                )
    except:
        alert.send_msg(msg='Ocorreu um error')
        
    alert.send_msg(msg='Inicio da execução')

    try:

        net.fit(dataset.train, dataset.test, steps=steps, view_callback=view_images, 
                send_update=send_update,
                error_save=save_error, time_wait=10000)
    except Exception as e :
        alert.send_msg(msg=f'Ocorreu um error durante o treinamento: {e}')

    try:
        
        net.generator.save('model_result.h5')
        ssim_vals.save_results(csv_file='./results_seach/ssim.csv')
        errors.save_results(csv_file='./results_seach/errors.csv')
        mse.save_results(csv_file='./results_seach/mse.csv')
        if net.pontuations != None:
            alert.send_msg(msg=f'for the final evaluation of the model we have:\nSSIM: {net.pontuations[0]}\nMSE: {net.pontuations[1]}\nPSNR: {net.pontuations[2]}')

    except:
        alert.send_msg(msg='Error in saved results')

    alert.send_msg(msg='Fim da execução')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('steps', type=int, help='an integer for total steps')
    parser.add_argument('loss_name', type=str, help='a string value')
    parser.add_argument('optimazer_name', type=str, help='a string value')
    parser.add_argument('lr', type=float, help='a float value')
    parser.add_argument('bt_1', type=float, help='a float value')
    parser.add_argument('bt_2', type=float, help='a float value')


    args = parser.parse_args()

    main(steps=args.steps, loss_name=args.loss_name, optimazer_name=args.optimazer_name, 
         lr=args.lr, bt_1=args.bt_1, bt_2=args.bt_2)