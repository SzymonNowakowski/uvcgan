### Author: Małgorzata Niemiec
### Executed: 2025/12/17

import os
import hashlib
import shutil

root_dir = "/data_hdd/kopia_crystals"  # searching root directory
dest_dir = "/data_hdd/unique_pngs"  # target directory for unique png files
os.makedirs(dest_dir, exist_ok=True)

# hash index function from file
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:       # open folder in binary mode
        while True:
            chunk = f.read(4096)       # load file in chunks
            if not chunk:
                break
            hash_md5.update(chunk)     # update hash with chunk
    return hash_md5.hexdigest()        # hash is returned as hex string


#  collect all png files with their md5 hashes
files_with_md5 = [] # list to store (file_path, md5) krotki
for dirpath, dirnames, filenames in os.walk(root_dir): # run through directory tree
    for f in filenames:
        if f.lower().endswith(".png"):
            full_path = os.path.join(dirpath, f)
            file_md5 = md5(full_path) # calculate md5 hash for png file
            files_with_md5.append((full_path, file_md5)) # adding (file_path, md5) to the list

# sort by md5
files_with_md5.sort(key=lambda x: x[1])

# copy unique files to destination directory
last_md5 = None
for full_path, file_md5 in files_with_md5:
    if file_md5 != last_md5:
        shutil.copy2(full_path, dest_dir)  # copy unique file with metadata (creation date, etc)
        last_md5 = file_md5
        last_path = full_path
    else:
        print(f"Duplicate of {last_path} skipped: {full_path}")

print(f"Finished, unique pngs saved in: {dest_dir}")
