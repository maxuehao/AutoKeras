import os, csv, cv2

from autokeras.image.image_supervised import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
 

#write csv  
def write_csv(img_dir, csv_dir):
    list = []
    list.append(['File Name','Label'])
    for file_name in os.listdir(img_dir):
    	for img in os.listdir("%s/%s"%(img_dir,file_name)):
            print (img)
            item = [file_name+"/"+img, file_name]
            list.append(item) 
    f = open(csv_dir, 'w')
    writer = csv.writer(f)
    writer.writerows(list)
 
#resize images
def resize_img(input_dir,output_dir):
    cls_file = os.listdir(input_dir)
    for cls_name in cls_file:
        img_file = os.listdir("%s/%s"%(input_dir,cls_name))
        for img_name in img_file:
            print (img_name)
            img = cv2.imread("%s/%s/%s"%(input_dir,cls_name,img_name))
            img = cv2.resize(img,(RESIZE,RESIZE),interpolation=cv2.INTER_LINEAR)
            if os.path.exists("%s/%s"%(output_dir,cls_name)):
                cv2.imwrite("%s/%s/%s"%(output_dir,cls_name,img_name),img)
            else:
                os.makedirs("%s/%s"%(output_dir,cls_name))
                cv2.imwrite("%s/%s/%s"%(output_dir,cls_name,img_name),img)


def train_autokeras(RESIZE_TRAIN_IMG_DIR,TRAIN_CSV_DIR,RESIZE_TEST_IMG_DIR,TEST_CSV_DIR,TIME):
    #Load images
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=RESIZE_TRAIN_IMG_DIR)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=RESIZE_TEST_IMG_DIR)

    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    print("Train data shape:", train_data.shape)

    clf = ImageClassifier(verbose=True)
    clf.fit(train_data, train_labels, time_limit=TIME)
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)

    y = clf.evaluate(test_data, test_labels)
    print("Evaluate:", y)

    #Predict the category of the test image
    img = load_img(PREDICT_IMG_PATH)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    x = np.reshape(x, (1, RESIZE, RESIZE, 3))
    print("x shape:", x.shape)

    y = clf.predict(x)
    print("predict:", y)

    clf.load_searcher().load_best_model().produce_keras_model().save(MODEL_DIR)

    #Save model architecture diagram
    model = load_model(MODEL_DIR)
    plot_model(model, to_file=MODEL_PNG)


if __name__ == "__main__":
    #Folder for storing training images
    TRAIN_IMG_DIR = '/media/pv/data2/cat_dog/train'
    RESIZE_TRAIN_IMG_DIR = '/media/pv/data2/cat_dog/resize_train'
    #Folder for storing testing images
    TEST_IMG_DIR = '/media/pv/data2/cat_dog/test'
    RESIZE_TEST_IMG_DIR = '/media/pv/data2/cat_dog/resize_test'

    #Path to generate csv file
    TRAIN_CSV_DIR = '/media/pv/data2/cat_dog/train_labels.csv'
    TEST_CSV_DIR = '/media/pv/data2/cat_dog/test_labels.csv'

    #Path to test image 
    PREDICT_IMG_PATH = 'resize_test/0/cat.4006.jpg'

    #Path to generate model file
    MODEL_DIR = 'Model.h5'
    MODEL_PNG = 'Model.png'

    #If your memory is not enough, please turn down this value.(my computer memory 16GB)
    RESIZE = 128
    #Set the training time, this is half an hour
    TIME = 0.5*60*60

    print ("Resize images...")
    resize_img(TRAIN_IMG_DIR,RESIZE_TRAIN_IMG_DIR)
    resize_img(TEST_IMG_DIR,RESIZE_TEST_IMG_DIR)
    print ("write csv...")
    write_csv(RESIZE_TRAIN_IMG_DIR, TRAIN_CSV_DIR)
    write_csv(RESIZE_TEST_IMG_DIR, TEST_CSV_DIR)
    print ("============Load...=================")
    train_autokeras(RESIZE_TRAIN_IMG_DIR,TRAIN_CSV_DIR,RESIZE_TEST_IMG_DIR,TEST_CSV_DIR,TIME)