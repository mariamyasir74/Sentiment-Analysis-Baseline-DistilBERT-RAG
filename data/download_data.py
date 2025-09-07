from datasets import load_dataset

def download_imdb(out_dir=None):
    ds = load_dataset('imdb')
    return ds

if __name__ == '__main__':
    download_imdb()
