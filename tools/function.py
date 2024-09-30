from os import path
import os
import shutil

def create_directory( ref):
    if path.exists(ref) == False:
        os.mkdir(ref)

def create_directory_recursive(root, dir_list):
    path = root

    for dir in dir_list:
        path = "{path}/{dir}".format(path=path, dir=dir)
        create_directory(path)

    return path

def copy_and_replace(source_path, destination_path):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)