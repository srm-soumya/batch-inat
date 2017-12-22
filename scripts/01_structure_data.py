import os
import pickle
import shutil
from PIL import Image
from collections import defaultdict

# Reduce the image size to (299, 299).
size = (299, 299)
data_path = 'data'
img_format = 'JPEG'

# Splits the data into training and validation set, you can choose the split-parameter.
split = 0.85
train = 'train'
test = 'validation'
cat_subcat = defaultdict(set)

def resize(img, size=size):
    '''
    Input: image, target_size
    Function: Resize the image to increase the training speed, or else the transformer would do this on the fly
    which might increase the training speed. This is a one time operation, you can choose to avoid this.
    Return: Resized Image, but keeps the aspect ratio
    '''
    img = img.resize(size, Image.ANTIALIAS)
    return img

# Goes through all the images in the specified data_path and resizes them
def resize_images():
    print('Resizing Images...')
    for path, dirs, files in os.walk(data_path):
        print(path)
        for file in files:
            name = os.path.join(path, file)
            img = Image.open(name)
            try:
                img = resize(img)
            except:
                print('Unable to resize image {}'.format(name))
            img.save(name, img_format)

def arrange_data():
    '''
    Move the images to their respective folder
    Data Structure change:
    data
        - cat
            - subcat
    data
        - train
            - cat-subcat
                - All the Images for this combination
        - validation
            - cat-subcat
                All the Images for this combination
    '''
    # Create train and validation directory
    os.makedirs(os.path.join(data_path, train))
    os.makedirs(os.path.join(data_path, test))

    print('Re-arranging data...')
    for cat in os.listdir(data_path):
        if cat not in {train, test}:
            print(cat)
            for sub_cat in os.listdir(os.path.join(data_path, cat)):
                cat_subcat[cat].add(sub_cat)
                folder_name = '{}-{}'.format(cat, sub_cat)

                # Create directories inside train and validation folders
                os.makedirs(os.path.join(data_path, train, folder_name))
                os.makedirs(os.path.join(data_path, test, folder_name))

                # Get the list of images
                imgs = os.listdir(os.path.join(data_path, cat, sub_cat))
                size = len(imgs)

                # Find the point to split
                split_point = int(split * size)

                # Move images to train directory
                for img in imgs[:split_point]:
                    src = os.path.join(data_path, cat, sub_cat, img)
                    dest = os.path.join(data_path, train, folder_name, img)
                    shutil.move(src, dest)

                # Move images to validation directory
                for img in imgs[split_point:]:
                    src = os.path.join(data_path, cat, sub_cat, img)
                    dest = os.path.join(data_path, test, folder_name, img)
                    shutil.move(src, dest)

            shutil.rmtree(os.path.join(data_path, cat))

    print('Saving the category-subcategory map in cat_subcat...')
    with open('metadata/cat_subcat', 'wb') as f:
        pickle.dump(cat_subcat, f)

if __name__ == '__main__':
    # Check if metadata exists or else create the directory
    if not os.path.exists('metadata'):
        os.makedirs('metadata')
    resize_images()
    arrange_data()
