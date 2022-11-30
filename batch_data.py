import gzip
import os
import multiprocessing as mp
import pickle
import argparse

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='data/example.json.gz', type=str,
                help="The input data file")    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = params()

    data_path = args.input
    data_folder, data_file = os.path.split(data_path)
    data_name = data_file.split('.')[0]
    batches_folder = os.path.join(data_folder, data_name)
    os.makedirs(batches_folder, exist_ok=True) 
    num_threads = mp.cpu_count()
    batch_size = 8192

    def get_next_filename(prefix='batch', suffix='pkl'):
        number = 0
        while True:
            number += 1
            yield '%s_%08d.%s' % (prefix, number, suffix)

    lines = []
    name_gen = get_next_filename(batches_folder+'/batch', 'json')

    with gzip.open(data_path, 'rt') as file_read:
        for line in file_read:
            lines.append(line)
            if len(lines) == batch_size:
                filepath = next(name_gen)
                with open(filepath, 'wt') as file_write:
                    file_write.writelines(lines)
                lines = []

    batch_files = os.listdir(batches_folder)
    batch_files = [ name for name in batch_files if '.json' in name ]
    batch_files.sort()
    batch_files = [ os.path.join(batches_folder, name) for name in batch_files ]
    len(batch_files)


    def itemize(batch_file):
        with open(batch_file, 'r') as file_read:
            lines = file_read.readlines()
        items =  [ eval(line.replace('true', 'True').replace('false', 'False')) for line in lines ] 
        new_name = batch_file.split('.')[0]+'.pkl'
        with open(new_name, 'wb') as file_write:
            pickle.dump(items, file_write)
        return len(items)

    with mp.Pool(num_threads) as p:
        items_batched_count = p.map(itemize, batch_files)


    batch_files = os.listdir(batches_folder)
    batch_files = [ name for name in batch_files if '.pkl' in name ]
    batch_files.sort()
    batch_files = [ os.path.join(batches_folder, name) for name in batch_files ]
    len(batch_files)

    for batch_file in batch_files:
        with open(batch_file, 'rb') as file_read:
            lines = pickle.load(file_read)
        break

    all_files = os.listdir(batches_folder)
    all_files = [ os.path.join(batches_folder, name) for name in all_files ]
    json_files = [ name for name in all_files if '.json' in name ]
    json_files.sort()
    pkl_files = [ name for name in all_files if '.pkl' in name ]
    pkl_files.sort()
    print(len(json_files), len(pkl_files))
    assert len(json_files) == len(pkl_files)


    with open(pkl_files[-1], 'rb') as file_read:
        lines = pickle.load(file_read)
        final_line_pkl = lines[0]
    print(final_line_pkl)

    with open(json_files[-1], 'r') as file_read:
        lines = file_read.readlines()
        final_line_json = lines[0]
    print(final_line_json)

    final_line_json = eval(final_line_json.replace('true', 'True').replace('false', 'False'))
    assert final_line_pkl == final_line_json

    print('passed:', data_file)