import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import multiprocessing as mp
import pickle
import math
import numpy as np
import time
import json
from collections import defaultdict
import tensorflow as tf
from tqdm import tqdm as progress_bar

from model import LatentFactorModel, trainingStep

def params():
  parser = argparse.ArgumentParser()

  parser.add_argument("--input", default='data/example.json.gz', type=str,
              help="The input data file")
  parser.add_argument("--split", default=100, type=int,
            help="test/train split ratio.")
  parser.add_argument("--factors", default=5, type=int,
            help="number of latent factors to use.")
  parser.add_argument("--epochs", default=1, type=int,
            help="Total number of training epochs to perform.")
  parser.add_argument("--learning-rate", default=0.1, type=float,
              help="Model learning rate starting point.")
  parser.add_argument("--regularization", default=0.0001, type=float,
              help="regularization factor.")

  args = parser.parse_args()
  return args

def count_entries(pkl_file):
    with open(pkl_file, 'rb') as file_read:
        items = pickle.load(file_read)
    return len(items)

def collect_ratings_data(pkl_file):
    with open(pkl_file, 'rb') as file_read:
        items = pickle.load(file_read)
        data = []
        for item in items:
            u,i,r = item['reviewerID'], item['asin'], item['overall']
            data.append((u,i,r))
        return data

  
if __name__ == "__main__":
  args = params()

  data_path = args.input
  data_folder, data_file = os.path.split(data_path)
  data_name = data_file.split('.')[0]
  batches_folder = os.path.join(data_folder, data_name)
  os.makedirs(batches_folder, exist_ok=True) 
  num_threads = mp.cpu_count()
  batch_size = 8192

  all_files = os.listdir(batches_folder)
  all_files = [ os.path.join(batches_folder, name) for name in all_files ]
  pkl_files = [ name for name in all_files if '.pkl' in name ]
  pkl_files.sort()
  # print(len(pkl_files))

  all_files = os.listdir(batches_folder)
  all_files = [ os.path.join(batches_folder, name) for name in all_files ]
  pkl_files = [ name for name in all_files if '.pkl' in name ]
  pkl_files.sort()
  # print(len(pkl_files))

  with mp.Pool(num_threads) as p:
    batch_lens = p.map(count_entries, pkl_files)
  dataset_len = sum(batch_lens)
  # print(dataset_len)

  start_time = time.time()
  with mp.Pool(num_threads) as p:
      datasets = p.map(collect_ratings_data, pkl_files)
  print("--- %s seconds ---" % (time.time() - start_time))
  # print(len(datasets))

  dataset_all = []
  for dataset in datasets:
      dataset_all.extend(dataset)
  # len(dataset_all)

  test_size = math.floor(len(dataset_all) / args.split)

  dataset_train = dataset_all[:-test_size]
  dataset_test = dataset_all[-test_size:]
  assert len(dataset_train)+len(dataset_test) == len(dataset_all)

  trainRatings = [r[2] for r in dataset_train]
  globalAverage = np.mean(trainRatings)
  print(globalAverage)

  for idx, item in enumerate(dataset_train):
      dataset_train[idx] = item[0], item[1], item[2] - globalAverage
  for idx, item in enumerate(dataset_test):
      dataset_test[idx] = item[0], item[1], item[2] - globalAverage


  ratingsPerUser = defaultdict(list)
  ratingsPerItem = defaultdict(list)
  userIDs = {}
  itemIDs = {}
  for u,i,r in dataset_train:
      ratingsPerUser[u].append((i,r))
      ratingsPerItem[i].append((u,r))
      if not u in userIDs: userIDs[u] = len(userIDs)
      if not i in itemIDs: itemIDs[i] = len(itemIDs)
  # with open('cache.pkl', 'wb') as cache_file:
  #     pickle.dump((ratingsPerUser, ratingsPerItem, userIDs, itemIDs), cache_file)
  # with open('cache.pkl', 'rb') as cache_file:
  #     ratingsPerUser, ratingsPerItem, userIDs, itemIDs = pickle.load(cache_file)

  mu = globalAverage

  u_test = []
  i_test = []
  r_actual = []
  for u,i,r in progress_bar(dataset_test, total=len(dataset_test)):
      if u not in userIDs or i not in itemIDs:
              continue
      else:
          u_test.append(userIDs[u])
          i_test.append(itemIDs[i])
          r_actual.append(r)
  u_test = np.array(u_test)
  i_test = np.array(i_test)
  r_actual = np.array(r_actual)

  optimizer = tf.keras.optimizers.Adam(args.learning_rate)
  model = LatentFactorModel(mu, args.factors, args.regularization, userIDs, itemIDs)

  # 10 iterations of gradient descent
  iterations = args.epochs
  pbar = progress_bar(range(iterations), total=iterations)

  results = []
  for i in pbar:
      obj = trainingStep(dataset_train, model, optimizer, userIDs, itemIDs)
      
      r_pred = model.predictSample(u_test, i_test).numpy()
      mse = np.mean(np.square(r_pred-r_actual))
      
      result = {'obj': obj, 'test:': mse}
      results.append(result)
      pbar.set_postfix(result)
      
  #     print("iteration " + str(i) + ", objective = " + str(obj))
  print("objective = " + str(obj))

  r_pred = model.predictSample(u_test, i_test).numpy()
  mse = np.mean(np.square(r_pred-r_actual))
  print(mse)

  baseline = np.mean(np.square( r_actual - globalAverage ))
  print(baseline)

  results_final = {
    'data_path': data_path,
    'batches_folder': batches_folder,
    'batch_size': batch_size,
    'args': vars(args),
    'results': results,
    'mse': mse,
    'baseline': baseline,
    'dataset_len': len(datasets),
  }

  results_folder = 'results'
  os.makedirs(results_folder, exist_ok=True) 
  results_file = os.path.join(results_folder, data_name+'.pkl')
  with open(results_file, 'wb') as fp:
    pickle.dump(results_final, fp)