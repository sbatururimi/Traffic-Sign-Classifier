
# coding: utf-8

# In[ ]:


import os
import zipfile
import shutil

from tqdm import tqdm
from urllib.request import urlretrieve

def _unzip(save_path, dataset_name, data_path):
    """
    Unzip wrapper
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """  
    print('Extracting {}...'.format(dataset_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract_dataset(dataset_name, data_path):
    """
    Download and extract the dataset
    :param dataset_name: Dataset name
    """
    DATASET_TRAFFIC_SIGNS_NAME = 'traffic-signs'
    if dataset_name == DATASET_TRAFFIC_SIGNS_NAME:
        url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip'
        extract_path = os.path.join(data_path, 'traffic-signs-data')
        save_path = os.path.join(data_path, 'traffic-signs-data.zip')
        extract_fn = _unzip
    else:
        print('Wrong dataset name')
    
    if os.path.exists(extract_path):
        print('Found {} Data'.format(dataset_name))
        return
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(dataset_name)) as pbar:
            urlretrieve(url,
                       save_path, 
                       pbar.hook)
            
    os.makedirs(extract_path)
    try:
        extract_fn(save_path, dataset_name, extract_path)
    except Exception as err:
        shutil.rmtree(extract_path) # remove extraction folder if there is an error
        raise err
        
    # Remove compressed data
    os.remove(save_path)
    
class DLProgress(tqdm):
    """
    Handle progress bar while downloading
    """
    last_block = 0
    
    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return 
            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

