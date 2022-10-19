import os 
import sys
import cv2 

#constants 

RESHAPE_SIZE = (64, 64)

#paths 

raw_data_path = 'data/raw_data'
retrieved_data_path = 'data/retrieved_data'
resized_data_path = 'data/resized_data'


#get list of files and process it

def getListOfFiles(dirname): 

    """
    Get list of files 
    
    """

    allFiles = list()

    for folder in next(os.walk(dirname)): 
        for file in os.listdir(os.path.join(dirname, folder)): 
            if file.endswith('.jpg'):
                allFiles.append(os.path.join(dirname, file))

   
    return allFiles
  


def createNewDataset(dirname):
    """
    create new dataset folder for retrieved files

    """

    allFiles = getListOfFiles(dirname)
    print(f'Number of files: {len(allFiles)}')

    dest_dir  = os.path.join(os.getcwd(), retrieved_data_path)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    print(dest_dir)
    i = 0 
    for file in allFiles:
        new_file_name = os.path.join(dest_dir, os.path.basename(file))
        os.system(f'cp {file} {new_file_name}')
        i += 1
        if i % 100 == 0:
            print(f'Processed {i} files')


#function to resize images

def resizeImages(dirname):

    """
    resize images to 64x64

    dirname: path to the directory containing images

    """
    i = 0

    try: 
        for image in os.listdir(dirname): 
            img = cv2.imread(os.path.join(os.getcwd(), dirname, image))
            if img is not None:
                resized_img = cv2.resize(img, RESHAPE_SIZE)
                cv2.imwrite(os.path.join(os.getcwd(), resized_data_path, image), resized_img)
                i += 1
                if i % 100 == 0:
                    print(f'Processed {i} files')
    except Exception as e:
        print(e)



if __name__ == '__main__':
    # createNewDataset(raw_data_path)
    resizeImages(retrieved_data_path)

   

