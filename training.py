import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
import numpy as np 
from model import architecture
from load_and_save import save_checkpoint, load_checkpoint
import torchvision
import cv2
# extracting data
def get_data():
    empty_images_path = 'clf-data\empty'
    empty_images = [os.path.join(empty_images_path,f) for f in os.listdir(empty_images_path)]
    empty_labels = [0.0]*len(empty_images)
    notempty_images_path = 'clf-data\\not_empty'
    notempty_image = [os.path.join(notempty_images_path,f) for f in os.listdir(notempty_images_path)]
    notempty_labels = [1.0]*len(notempty_image)
    images = notempty_image + empty_images
    labels = notempty_labels + empty_labels

    combine_images_labels = list(zip(images,labels))
    random.shuffle(combine_images_labels)
    images,labels = zip(*combine_images_labels)


    training_ratio = 0.9
    training_images,training_labels = images[:int(len(images)*training_ratio)],labels[:int(len(labels)*training_ratio)]
    testing_images,testing_labels = images[int(len(images)*training_ratio):],labels[int(len(labels)*training_ratio):]
    return training_images,training_labels,testing_images,testing_labels


LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "weights/model_car_parking.pth.tar"
SAVE_MODEL_FILE = "weights/model_car_parking.pth.tar"



class Data(Dataset):
    def __init__(self,images,labels,istransform=True) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.istransform = istransform  
        self.transforms = transforms.Compose([transforms.Resize((32,69)),transforms.ToTensor()]) 
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.istransform:
            image = self.transforms(image)
        return image,label





def accuracy(predict,actual,threshold=0.5):
    predict = predict.detach().cpu().tolist()
    predict = np.array([1.0 if i >= threshold else 0.0 for i in predict])
    actual = actual.detach().cpu().numpy()
    return sum(predict == actual)/len(predict)


def train(model : torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        train_iterator: DataLoader,
        valid_iterator: DataLoader,
        save_path: str,
        Device= "cpu",
        epochs=10):

    model.to(Device)
    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)
        training_loss = []
        accuracy_train = []
        # training_loss2 = []
        loss_val = 0  
        # set training mode
        model.train()
        # Loop through the training batch
        for batch, (image,label) in enumerate(train_iterator):
            image = image.to(Device)
            label = label.to(Device)
            predict_value = model(image)
            # computing loss
            loss = loss_fn(predict_value.squeeze(),label.float())
            # loss2 = loss_fn1(predict_value,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            training_loss.append(loss.item())
            acc =accuracy(predict_value.squeeze(),label.float())
            accuracy_train.append(acc)
            # training_loss2.append(loss2.item())
            pbar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)},accuracy_training = {round(acc,3)}", refresh=False)
            pbar.update()
        # ------ inference time --------
        loss_val = 0
        accuracy_val = []
        with torch.inference_mode():
            # Set the model to eval
            model.eval()
            # Loop through the validation batch
            for batch,(image,label) in enumerate(valid_iterator):
                image = image.to(Device)
                label = label.to(Device)
                y_predict_inference = model(image)
                loss_val += loss_fn(y_predict_inference.squeeze(),label.float()).item()
                accuracy_val.append(accuracy(y_predict_inference.squeeze(),label.float()))
        pbar.set_postfix(
            epoch=f" {epoch},train loss= {round(sum(training_loss) / len(training_loss), 4)}, val loss = {round(loss_val/len(valid_iterator),3)},training_accuracy = {round(sum(accuracy_train)/len(accuracy_train),3)}, val_accuracy = {round(sum(accuracy_val)/len(accuracy_val),3)}",refresh=False)
        pbar.close()
        if epoch % 1 == 0:
            check_point = {
                "state_dict": model.state_dict(),}
            save_checkpoint(check_point,save_path)
    return model





def main():
    training_images,training_labels,testing_images,testing_labels = get_data()
    training_data = Data(training_images,training_labels,istransform=True)
    valid_data = Data(testing_images,testing_labels,istransform=True)
    train_iterator = DataLoader(training_data,batch_size=BATCH_SIZE,shuffle=True)
    valid_iterator = DataLoader(valid_data,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    model = architecture()

    loss = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model)
    
    model = train(model,loss,optim,train_iterator,valid_iterator,SAVE_MODEL_FILE)
    
    # data0 = iter(valid_iterator)
    # image0,label0,name_iamges = next(data0)
    # print(image0.shape)
    # predict_value = model(image0)
    # print(name_iamges)
    # print(accuracy(predict_value.squeeze(),label0[0:1].float()))
    # print(predict_value)
    # print(label0)

    return

if __name__ == "__main__":
    main()