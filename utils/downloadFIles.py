import os
import zipfile

def download_unzip_dataset (file_id, path):
    save_path = os.path.join(os.getcwd(), 'data', 'raw_data')
    output = os.path.join(save_path, 'wikiart.zip')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.system(' curl -L -o {} {}'.format(path, file_id))
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(save_path)


if __name__=='__main__':
    file_id = 'http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip'
    path = 'data/raw_data/wikiart.zip'
    download_unzip_dataset(file_id, path)
