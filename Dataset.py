import os
from os import path, listdir

from torchvision.models import inception_v3
from torchvision.transforms import ToTensor

from tqdm import tqdm

from fastai.vision.all import PILImage

from pickle import dump, load

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, add
from tensorflow.keras.models import Model, load_model

from fastai.callback.all import *
from fastai.callback.progress import *

from torch import no_grad
import matplotlib.pyplot as plt

import random




from numpy import array, argmax

import warnings
warnings.filterwarnings("ignore")


class Dataset:
    model = inception_v3(pretrained=True)
    features = {}
    captions = {}
    all_captions = []
    train_imgs = []
    test_imgs = []
    epochs = 11
    batch_size = 64

    def save_features(self):
        with open("features.pkl", "wb") as f:
            dump(self.features, f)

    def get_features(self):
        with open("features.pkl", "rb") as f:
            self.features = load(f)

    def extract_features(self):
        for image in tqdm(listdir(self.img_path)):
            img = PILImage.create(path.join(self.img_path, image))
            img = img.resize((299, 299))
            img = ToTensor()(img)
            img = img.unsqueeze(0)

            self.model.eval()                                           
            with no_grad(): 
                self.features[image] = self.model(img).numpy()   
        
            

    def get_captions(self):
        with open(self.caption_path, "r") as f:
            self.captions_unformatted = f.read()
        self.split_captions()

    def get_all_captions(self):
        for key, val in self.captions.items():
            for cap in val:
                self.all_captions.append(cap)

    def split_captions(self):
        for line in self.captions_unformatted.split('\n'):      #split the captions into lines
            parts = line.split(',')            #split the lines into parts
            if len(line) < 2:                  #if the line is less than 2 parts, skip
                continue

            image_id, image_caption = parts[0], parts[1:] 
            image_id = image_id.split('.')[0]
            image_caption = ' '.join(image_caption)


            image_caption = image_caption.lower()
            image_caption = image_caption.replace('.', '')
            image_caption = 'startseq ' + image_caption + ' endseq'

            if image_id not in self.captions:         #collate the captions for each image
                self.captions[image_id] = list()
            
            self.captions[image_id].append(image_caption)

            
    def create_tokens(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.all_captions)
        self.tokens = self.tokenizer 

    def create_test_train(self):

        for i in range(1, 6000):
            self.train_imgs.append(list(self.captions.keys())[i])

        for j in range(6000, 8000):
            self.test_imgs.append(list(self.captions.keys())[j])



    def create_sequences(self, tokenizer, max_length, desc_list, image):
        X1, X2, y = list(), list(), list()
        vocab_size = len(tokenizer.word_index) + 1
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(image)
                X2.append(in_seq)
                y.append(out_seq)
                
        return array(X1), array(X2), array(y)
    
    def define_model(self, vocab_size, max_length):
        inputs1 = Input(shape=(1000,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def generate_data(self, descriptions, photos, tokenizer, max_length):
        while True:
            for key, desc_list in descriptions.items():
                if key == "image":
                    continue

                photo = photos[key+".jpg"][0]
                in_img, in_seq, out_word = self.create_sequences(tokenizer, max_length, desc_list, photo)
                yield [[in_img, in_seq], out_word]

    def generate_desc(self, tokenizer, photo, max_length):
        in_text = 'startseq'
        photo = photo.reshape((1, 1000))

        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = self.loaded_model.predict([photo, sequence], verbose=0)
            yhat = argmax(yhat)
            word = tokenizer.index_word[yhat]
            if word == 'endseq':
                break
            in_text += ' ' + word
        return in_text
    
    def train_model(self):
        steps = len(self.train_imgs)
        for i in range(self.epochs): 
            generator = self.generate_data(self.captions, self.features, self.tokens, 58)
            self.model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            self.model.save('model.h5')

    def test_single(self):
        pass




    def __init__(self, img_path, caption_path):
        self.img_path = img_path
        self.caption_path = caption_path
        self.captions = {}
        self.captions_unformatted = ""
        self.all_captions = []
        self.train_imgs = []
        self.test_imgs = []
        self.features = {}
        self.tokens = None
        self.model = None
        self.epochs = 10

    def test_model(self):
        self.loaded_model = load_model("model.h5")
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')
        r = random.randint(0, 1995)

        image = self.train_imgs[r]
        output = self.generate_desc(self.tokenizer, self.features[image + ".jpg"][0], 58)
        output = output.replace("startseq", "")
        print("Actual: ", self.captions[image])
        print("Predicted: ", output)
        plt.imshow(plt.imread(self.img_path + "/" + image + ".jpg"))
        plt.title("predicted: " + output)
        plt.show()
        

        



        # for i in range(r, r + 5):
        #     image = self.test_imgs[i]
        #     file = self.img_path + "/" + image + ".jpg"

        #     photo = self.features[image + ".jpg"][0]
        #     description = self.generate_desc(self.tokenizer, photo, 58)
        #     description = description.replace("startseq", "")
        #     print("Actual: ", self.captions[image])
        #     print("Predicted: ", description)
            
        #     plt.imshow(plt.imread(file))
        #     plt.title("predicted: " + description)
        #     plt.show()


    def __init__(self, path, extract_features, train, test):
        self.overall_path = path
        self.img_path = os.path.join(self.overall_path, "Images")
        self.caption_path = os.path.join(self.overall_path, "captions.txt")

        if extract_features:
            self.extract_features()
            self.save_features()

        self.get_features()
        
        if train:
            self.get_captions()
            self.get_all_captions()
            self.create_tokens()
            self.create_test_train()
            self.define_model(len(self.tokenizer.word_index) + 1, 58)
            self.train_model()

        if test:
            self.get_captions()
            self.get_all_captions()
            self.create_tokens()
            self.create_test_train()
            self.test_model()
   
