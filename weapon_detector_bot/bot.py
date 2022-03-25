from itertools import count
import telebot
import os
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pickle
import telebot
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import time
from torch import nn
import uuid
import matplotlib.pyplot as plt

class ResNet101(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        
        self.model.fc = nn.Linear(2048, 10)
        self.linear = nn.Linear(10, 2)
        
        layers_count = len(list(self.model.parameters()))
        for i, parameter in enumerate(self.model.parameters()):
            if i < layers_count - 5:
                parameter.requires_grad = False
                
                
    def forward(self, X):
        logits = self.model(X)
        if self.training:
            logits = self.linear(logits)
        return logits



class ResNet152(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)
        
        self.model.fc = nn.Linear(2048, 10)
        self.linear = nn.Linear(10, 2)
        
        layers_count = len(list(self.model.parameters()))
        for i, parameter in enumerate(self.model.parameters()):
            if i < layers_count - 10:
                parameter.requires_grad = False
                
                
    def forward(self, X):
        logits = self.model(X)
        if self.training:
            logits = self.linear(logits)
        return logits


class DenseNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=pretrained)
        
        self.model.classifier = nn.Linear(1920, 10)
        self.linear = nn.Linear(10, 2)
        
        layers_count = len(list(self.model.parameters()))
        for i, parameter in enumerate(self.model.parameters()):
            if i < layers_count - 5:
                parameter.requires_grad = False
                
                
    def forward(self, X):
        logits = self.model(X)
        if self.training:
            logits = self.linear(logits)
        return logits

transform = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(350),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



MODEL_FOLDER = './models/stacking/'


resnet152 = ResNet152()
resnet152.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'ResNet152.pt')))
resnet152.eval();

resnet101 = ResNet101()
resnet101.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'ResNet101.pt')))
resnet101.eval();

densenet = DenseNet()
densenet.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'DenseNet.pt')))
densenet.eval();

boosting = CatBoostClassifier().load_model(os.path.join(MODEL_FOLDER, 'boosting.model'))
random_forest = pickle.load(open(os.path.join(MODEL_FOLDER, 'random_forest.pkl'), 'rb'))
svm = pickle.load(open(os.path.join(MODEL_FOLDER, 'svm.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_FOLDER, 'scaler.pkl'), 'rb'))


def get_prediction(frame):
    WEIGHTS = np.array([1, 1.32])
    transformed_image = transform(frame)
    
    resnet101_logits = scaler.transform(resnet101(transformed_image[None,]).detach().numpy())
    resnet152_logits = resnet152(transformed_image[None,]).detach().numpy()
    densenet_logits = scaler.transform(densenet(transformed_image[None,]).detach().numpy())
    #print(resnet152_logits)
    #plt.imshow(resnet152_logits, interpolation='nearest')
    #plt.show()
    predictions = np.array([0, 0])
    
    boosting_resnet101_probas = boosting.predict_proba(resnet101_logits)
    boosting_resnet152_probas = boosting.predict_proba(resnet152_logits)
    boosting_densenet_probas = boosting.predict_proba(densenet_logits)
    
    forest_resnet101_probas = random_forest.predict_proba(resnet101_logits)
    forest_resnet152_probas = random_forest.predict_proba(resnet152_logits)
    forest_densenet_probas = random_forest.predict_proba(densenet_logits)
    
    svm_resnet101_probas = svm.predict_proba(resnet101_logits)
    svm_resnet152_probas = svm.predict_proba(resnet152_logits)
    svm_densenet_probas = svm.predict_proba(densenet_logits)
    
    predictions = (boosting_resnet101_probas + boosting_densenet_probas + 
                forest_resnet101_probas + forest_densenet_probas + 
                svm_resnet101_probas + svm_densenet_probas + 
                boosting_resnet152_probas + forest_resnet152_probas +
                svm_densenet_probas) / 4
    predictions = predictions * WEIGHTS
    predictions /= predictions.sum()
    return predictions



bot = telebot.TeleBot('5280669995:AAHTZAWdOQnMvghD-2-GYnXgtILliiePcRI')
@bot.message_handler(commands=["start"])
def start(m, res=False):
        # Добавляем две кнопки
        markup=telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1=telebot.types.KeyboardButton("Отслеживать")
        item2=telebot.types.KeyboardButton("Стоп")
        markup.add(item1)
        markup.add(item2)
        bot.send_message(m.chat.id, 
                        'Нажми: \n<u>Отслеживание</u> для начала сессии' + 
                        '\n<u>Стоп</u> для остановки сессии', 
                        reply_markup=markup, 
                        parse_mode='HTML')

start, calls = 0, 0
@bot.message_handler(content_types=["text"])
def handle_text(message):
    global start, calls
    if message.text.strip() == 'Стоп':
        answer = 'Сессия окончена. \n<b>Статистика:</b>' + f'\n<u>Время сессии</u>: {int((time.time() - start)/60)} минут'
        answer += f'\n<u>Количество оповещений</u>: {calls}'
        bot.send_message(message.chat.id, answer, parse_mode='HTML')
        bot.stop_bot()
    # Если юзер прислал 1, начало отслежки
    if message.text.strip() == 'Отслеживать' :
        start = time.time()
        calls = 0 # Positive detection
        prev_call = False
        call = False 
        it = 0
        counter = [False] * 10
        count_true_calls = 0# All detections before positive detection
        bot.send_message(message.chat.id, 'Старт')
        bot.send_message(message.chat.id, 'Подключение к камере:')
        bot.send_chat_action(message.chat.id, action='typing', timeout=10000)
        video_capture = cv2.VideoCapture(0)
        bot.send_message(message.chat.id, 'Подключение к камере успешно!')
        while message.text.strip() != 'Стоп':
            _, frame = video_capture.read()
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            if it == 9:
                it = 0
            it += 1
            cv2.imshow('Camera', frame)
            probas = get_prediction(frame)
            predicted_class = np.argmax(probas) if np.max(probas) > 0.63 else 0
            print('Predicted class is: ', predicted_class, " with probas ", probas)
            counter[it] = True if predicted_class == 1 else False
            if predicted_class == 1:
                print('Probas: ', probas)
                call = True
            else:
                call = False
            if prev_call == True and call == True:
                count_true_calls += 1
            else:
                count_true_calls = 0
            print(count_true_calls)
            if count_true_calls > 1 or sum(counter) > 5:
                print(count_true_calls)
                print(counter)
                answer = 'Внимание, возможно, в кадре оружие.\nКадр:'
                bot.send_chat_action(message.chat.id, 'upload_photo')
                bot.send_message(message.chat.id, answer)     
                uuid_str = uuid.uuid4().hex
                TMP_ATTENTION_FOLDER = './attention/tmpfile_%s.jpg' % uuid_str     
                cv2.imwrite(TMP_ATTENTION_FOLDER, frame)                      
                bot.send_photo(message.chat.id, photo=open(TMP_ATTENTION_FOLDER, 'rb'))
                os.remove(TMP_ATTENTION_FOLDER)
                counter = [False] * 10
                count_true_calls = 0
                calls += 1
            prev_call = call


print('Всё готово для начала вакханалии')
bot.polling(none_stop=True, interval=0)