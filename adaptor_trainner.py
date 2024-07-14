# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import subprocess
from IPython import display
from modules.msg_alert import MsgAlert

"""# Genetic Algorithm

Algoritmo genetico para otimizar os paramentros
"""

class Individual:
  chromosome = None
  pontuation = None

  def __init__(self, chromosome=None, bits=8):
    self.bits = bits
    self.max_val = 2**bits-1
    if chromosome!=None:
      self.chromosome = chromosome
    else:
      self.chromosome = random.randint(0, self.max_val)

    self.pontuation = 0.0

  def muted(self,chromosome_result):

    # Flip value with 1
    mask_mut = random.randint(1,self.max_val)
    maks = 0b1

    for _ in range(self.bits):
      if mask_mut & 0b1:
        chromosome_result = chromosome_result ^ maks

      mask_mut = maks << 1
      mask_mut = mask_mut >> 1

    return chromosome_result

  def cross(self,femele):
    prob_muted_fathers = random.random()
    male = self.chromosome
    femele = femele.chromosome

    if prob_muted_fathers < 0.2:
      male = self.muted(male)
    elif prob_muted_fathers < 0.4:
      femele = self.muted(femele)

    mask_mut = random.randint(1,self.max_val)
    child = 0
    maks = 0b1 << self.bits

    for _ in range(self.bits):
      if mask_mut & 0b1:
        child = child | (male & maks)
      else:
        child = child | (femele & maks)

      maks = maks >> 1
      mask_mut = mask_mut >> 1

    return Individual(chromosome=child, bits=self.bits)

  def get_indice(self,mapped_bits=2):
    list_indices = []
    mask = 2**mapped_bits-1
    chromo = self.chromosome

    for _ in range(self.bits//mapped_bits):
      val = chromo & mask
      list_indices.append(val)
      chromo = chromo >> mapped_bits

    list_indices.reverse()

    return list_indices


  def __str__(self,):
    return f"chromo: {bin(self.chromosome)}, pont: {self.pontuation}"

  def __lt__(self, individual_comparator):
    return self.pontuation > individual_comparator.pontuation

  def __eq__(self, __o: object) -> bool:
    val = self.chromosome == __o.chromosome
    return val


"""## Definindo os paramentros"""
alert = MsgAlert()
loss_functions = ['mse_loss','ssim_loss','ssim_mse_loss','ssim_psnr_mse_loss','mse_loss','ssim_loss','ssim_mse_loss','acurracy_kl_loss',]
opt_names = ['Adam','AdamX', 'RMSprop', 'SGD','Adam','AdamX', 'RMSprop', 'SGD']
lrs_gen = [0.0001, 0.001 , 0.01  , 0.1, 0.00046415888336127773, 0.002154434690031882, 0.02  , 0.2   ]
bts_1_gen = [0.2,0.9,0.95,0.85, 0.2,0.9,0.95,0.85]
bts_2_gen = [0.9999,0.995,0.999,0.99,0.9999,0.995,0.999,0.99]

total_individous = 10
population = np.array([ Individual( bits=15 ) for _ in range(total_individous) ]) if not os.path.isfile('./population.npy') else np.load('./population.npy', allow_pickle=True)
generations = 2
steps = 100

for generation in range(generations+1):

  alert.send_msg(msg=f'Generation:{generation}')

  # calculando a pontuação
  for i in range(total_individous):

    individou = population[i]
    indices = individou.get_indice(mapped_bits=3)

    # Command to execute
    command = f"python trainner_model.py 100 {loss_functions[indices[0]]} {opt_names[indices[1]]} {lrs_gen[indices[2]]} {bts_1_gen[indices[3]]} {bts_2_gen[indices[4]]} 1"

    # Execute the command and wait for it to complete
    process = subprocess.run(command, shell=True)

    # Check if the command was executed successfully
    print('.', end='', flush=True)
    if process.returncode == 0:
        try:
          pontuation = np.load(f"./outputs/1.npy")
        except:
          pontuation = 100.0

        individou.pontuation = pontuation
    else:
        print(f"Execution failed with return code {process.returncode}")


  population = np.sort(population)[::-1]

  individou = random.choice(population[:3]) 
  indices = individou.get_indice(mapped_bits=3)

  # Command to execute
  command = f"python training.py 100000 {loss_functions[indices[0]]} {opt_names[indices[1]]} {lrs_gen[indices[2]]} {bts_1_gen[indices[3]]} {bts_2_gen[indices[4]]}"

  # Execute the command and wait for it to complete
  process = subprocess.run(command, shell=True)

  with open('./population.npy', 'wb') as f:
    np.save(f,population)

  if generation >= generations:
    break

  new_generation = []

  s = int(total_individous*0.1) if int(total_individous*0.1) != 0 else 1

  new_generation.extend(population[:s])

  population_for_adaptar = int(total_individous*0.9)

  for _ in range(population_for_adaptar):


    male = random.choice(population[:int(total_individous*0.95)])
    femele = random.choice(population[:int(total_individous*0.95)])

    child = male.cross(femele)
    if not (child in new_generation):
      new_generation.append(child)
    else:
      cont = 0

      while child in new_generation:

        male = random.choice(population[:int(total_individous*0.95)])
        femele = random.choice(population[:int(total_individous*0.95)])

        child = male.cross(femele)
        
        if cont > 5000:
          break

        cont += 1

      new_generation.append(child)


  population = new_generation

  display.clear_output(wait=True)

  indices = population[0].get_indice()
  alert.send_msg(msg=f"python training.py  100000 {loss_functions[indices[0]]} {opt_names[indices[1]]} {lrs_gen[indices[2]]} {bts_1_gen[indices[3]]} {bts_2_gen[indices[4]]}")

alert.send_msg(msg='Adaptor trainner: fim da execução')