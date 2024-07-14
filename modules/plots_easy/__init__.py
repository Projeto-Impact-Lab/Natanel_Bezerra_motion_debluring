import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PlotsEasy:

    def __init__(self, xlabel='Epochs', ylabel='Loss', title='Loss X Epoch') -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title  = title
        self.epochs = list()
        self.loss_vals = list()

    def append(self, epoch, val_erro):
        self.epochs.append(epoch)
        self.loss_vals.append(val_erro)

    def plot(self, ):
        plt.plot(self.epochs, self.loss_vals, linestyle='-')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

    def save(self, path='figure.png'):
        plt.figure(figsize=(15, 15))
        plt.plot(self.epochs, self.loss_vals, linestyle='-')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)
        plt.savefig(path)
        plt.clf()
        plt.close()

    def save_results(self,csv_file='data.csv'):
        try:
            # Save the data to the CSV file
            pd.DataFrame({self.xlabel: self.epochs, self.ylabel: self.loss_vals}).to_csv(csv_file, index=False)
            return True
        except:
            return False

