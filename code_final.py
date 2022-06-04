#%%
#IMPORT THƯ VIỆN KERAS VÀ TENSOFLOW
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd 
import numpy as np 
import os 

from tensorflow import keras 

#%%
###SỬ DỤNG THƯ VIỆN ResNet50 CỦA CNN ĐỂ BIẾN ĐỔI DATA IMAGE THÀNH VECTOR

#GỌI THƯ VIỆN RESNET50 VÀ TRUYỀN TẬP DỮ LIỆU

from keras.applications.resnet import ResNet50
resnet50 = ResNet50(weights='imagenet', include_top=False)

def _get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resnet_features = resnet50.predict(img_data)
    return resnet_features

basepath = "fruits-360/Training/"
class1 = os.listdir(basepath + "Banana/")
class2 = os.listdir(basepath + "Strawberry/")
class3 = os.listdir(basepath + "Cocos/")
class4 = os.listdir(basepath + "Kiwi/")
class5 = os.listdir(basepath + "Lemon/")
class6 = os.listdir(basepath + "Cherry 1/")
class7 = os.listdir(basepath + "Kaki/")
class8 = os.listdir(basepath + "Lychee/")
class9 = os.listdir(basepath + "Mango/")
class10 = os.listdir(basepath + "Dates/")
class11 = os.listdir(basepath + "Orange/")
class12 = os.listdir(basepath + "Papaya/")
class13 = os.listdir(basepath + "Physalis/")
class14 = os.listdir(basepath + "Pepino/")
class15 = os.listdir(basepath + "Plum/")
class16 = os.listdir(basepath + "Salak/")
class17 = os.listdir(basepath + "Fig/")
class18 = os.listdir(basepath + "Walnut/")
class19 = os.listdir(basepath + "Pear/")
class20 = os.listdir(basepath + "Tomato 1/")




#file data gồm 3 tập , tập dữ liệu cho trái cây , và tập cho test 
data = {'banana': class1[:10], 
        'strawberry': class2[:10], 'Cocos': class3[:100],
        'Kiwi': class4[:100], 'Lemon': class5[:100],
        'Cherry 1' : class6[:100], 'Kaki': class7[:100],
        'Lychee' : class8[:100], 'Mango': class9[:100],
        'Dates' : class10[:100], 'Orange' : class11[:100],
        'Papaya' : class12[:100], 'Physalis' : class13[:100],
        'Pepino' : class14[:100], 'Plum' : class15[:100],
        'Salak' : class16[:100], 'Fig': class17[:100],
        'Walnut' : class18[:100], 'Pear' : class19[:100],
        'Tomato 1' : class20[:100], 
        'test': [class1[11], class2[11], class3[11], class4[11], class5[11]
        , class6[11], class7[11], class8[11], class9[11], class10[11] , class11[11] ,
        class12[11], class13[11], class14[11], class15[11], class16[11], class17[11] 
        , class18[11], class19[11], class20[11]]}

# %%

#BIẾN ĐỔI ẢNH THÀNH VECTOR
features = {"banana" : [], "strawberry" : [],'Cocos' : [] ,'Kiwi' :[],'Lemon' : [] , 
            "Cherry 1" : [], "Kaki" : [] , "Lychee" : [], "Mango" : [] , "Dates" : []
            ,"Orange" : [], "Papaya" : [], "Physalis" : [], "Pepino" : [],
            "Plum" :[] , "Salak" : [], "Fig" :  [], "Walnut" :[] , "Pear" :[],"Tomato 1" :[]
            ,"test" : [] }
testimgs = []
for label, val in data.items():
    for k, each in enumerate(val):        
        if label == "test" and k == 0:
            img_path = basepath + "/Banana/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 1:
            img_path = basepath + "/Strawberry/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 2:
            img_path = basepath + "/Cocos/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 3:
            img_path = basepath + "/Kiwi/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 4:
            img_path = basepath + "/Lemon/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 5:
            img_path = basepath + "/Cherry 1/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 6:
            img_path = basepath + "/Kaki/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 7:
            img_path = basepath + "/Lychee/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 8:
            img_path = basepath + "/Mango/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 9:
            img_path = basepath + "/Dates/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 10:
            img_path = basepath + "/Orange/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 11:
            img_path = basepath + "/Papaya/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 12:
            img_path = basepath + "/Physalis/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 13:
            img_path = basepath + "/Pepino/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 14:
            img_path = basepath + "/Plum/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 15:
            img_path = basepath + "/Salak/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 16:
            img_path = basepath + "/Fig/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 17:
            img_path = basepath + "/Walnut/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 18:
            img_path = basepath + "/Pear/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 19:
            img_path = basepath + "/Tomato 1/" + each
            testimgs.append(img_path)
        else: 
            img_path = basepath + label.title() + "/" + each
        feats = _get_features(img_path)
        features[label].append(feats.flatten()) 

#%%: BIỂN ĐỔI ẢNH
dataset = pd.DataFrame()
for label, feats in features.items():
    temp_df = pd.DataFrame(feats)
    temp_df['label'] = label
    dataset = dataset.append(temp_df, ignore_index=True)
dataset.head()


#CHIA TẬP DỮ DATA ĐƯỢC BIẾN ĐỔI THÀNH FILE X và y (target)

y = dataset[dataset.label != 'test'].label
X = dataset[dataset.label != 'test'].drop('label', axis=1)
# %%
###VIẾT THUẬT TOÁN CLASSIFIER ĐỂ HỌC 2 CLASS ĐƯỢC CHIA 

from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

model = MLPClassifier(hidden_layer_sizes=(100, 10))
pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
pipeline.fit(X, y)

print ("Model Trained on pre-trained features")

#PREDICT ẢNH TRONG FILE TEST SAU KHI THUẬT TOÁN HỌC ĐƯỢC

preds = pipeline.predict(features['test'])

f, ax = plt.subplots(1, 8, figsize=(30,30))
for i in range(8):
    ax[i].imshow(Image.open(testimgs[i]).resize((200, 200), Image.ANTIALIAS))
    ax[i].text(10, 180, 'Predicted: %s' % preds[i], color='k', backgroundcolor='red', alpha=0.8)
plt.show()