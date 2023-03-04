from PIL import Image
import os
import shutil

old_dir1 = './datasets/Nordland/query/'
old_dir2 = './datasets/Nordland/ref/'
new_dir1 = './datasets/Nordland_28x28/query/'
new_dir2 = './datasets/Nordland_28x28/ref/'
extension = '.jpg'

img_dirs = [old_dir1, old_dir2]
new_img_dirs = [new_dir1, new_dir2]

new_size = (28,28)

n_old2 = 27592 
n_old1 = 2760
n_imagess = [n_old1, n_old2]

if os.path.isdir('./datasets/Nordland_28x28'):
    pass
else:
    os.makedirs('./datasets/Nordland_28x28')

src_gt = './datasets/Nordland/ground_truth_new.npy'
dst_gt = './datasets/Nordland_28x28/ground_truth_new.npy' 

shutil.copyfile(src_gt, dst_gt)

for j in range(len(img_dirs)):
    img_dir = img_dirs[j]
    new_img_dir = new_img_dirs[j]
    n_images = n_imagess[j]
   
    if os.path.isdir(new_img_dir[:-1]):
        pass
    else:
        os.makedirs(new_img_dir[:-1])

    for i in range(n_images):
        temp_img = Image.open(img_dir+str(i).zfill(7)+extension)
        new_img = temp_img.resize(new_size) 
        new_img.save(new_img_dir+str(i).zfill(7)+'.jpg', 'JPEG', optimize=True)

