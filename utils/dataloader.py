import os

"""
Load all files in the data folders and subfolders into a single folder for easier processing.

"""

def getListOfFiles(dirname):
    """
    Get all files in a directory and subdirectories.
    """
    listOfFile = os.listdir(dirname)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirname, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def main():

    folder_name = 'data'
    dirname = os.path.join(os.getcwd(), folder_name)

    allFiles = getListOfFiles(dirname)

    for elem in allFiles:
        print(elem)

    print("---------------------")
    print("Total files: {}".format(len(allFiles)))

    # Create a new folder to store the data
    new_folder_name = 'retrieved_data'
    new_dirname = os.path.join(os.getcwd(), new_folder_name)
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)


    for elem in allFiles:
        # Copy the file to the new folder
        new_file_name = os.path.join(new_dirname, os.path.basename(elem))
        print(new_file_name)
        os.system('cp {} {}'.format(elem, new_file_name))


if __name__ == "__main__":
    main()