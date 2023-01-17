import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import torch
from PIL import Image
import random
import torch
# from torchvision.utils import draw_bounding_boxes
# import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.io import read_image
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, accuracy_score


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
class NFLDataset(torch.utils.data.Dataset):
    def __init__(self,labels, helmets, tracking,transforms, img_path):
        self.labels = labels
        self.helmets = helmets
        self.tracking = tracking
        self.img_path = img_path
        self.transforms = transforms

    def __getitem__(self, idx):
        views = ['Endzone', 'Sideline']
        label_record = self.labels.iloc[idx]
        label = label_record['contact'].astype(np.float32)
        game_play = label_record['game_play']
        frame = int(label_record['frame'])
        # boxes =[]
        features =[]
        for player in [1,2]:

            player_id = label_record[f"nfl_player_id_{player}"]
            if player_id!="G":
                player_id = int(player_id)


                player_helmet_record = self.helmets[(self.helmets['game_play']==game_play)&
                (self.helmets['nfl_player_id']==player_id)&(self.helmets['frame']==frame)]
                player_tracking_record = self.tracking[(self.tracking['game_play']==game_play)&
                (self.tracking['nfl_player_id']==player_id)&(self.tracking['frame']==frame)]
                if len(player_tracking_record)>0:
                    player_tracking_record = player_tracking_record[['x_position', 'y_position', 'speed', 'direction', 'orientation', 'acceleration', 'sa']].values[0]
                else:
                    player_tracking_record = np.array([0,0,0,0,0,0,0])
                if len(player_helmet_record)>0:
                    player_helmet_record= player_helmet_record[["top", "left", "height", "width"]].values[0]
                else:
                    player_helmet_record = np.array([0,0,0,0])
                feature = np.concatenate([player_tracking_record, player_helmet_record])

                features.append(torch.FloatTensor(feature))
            else:
                features.append(torch.FloatTensor([0,0,0,0,0,0,0,0,0,0,0]))

        
        for view in views:
            file_name = game_play+"_" + view +".mp4_{:04d}.jpg".format(frame)
            image = read_image(self.img_path+file_name)
            if view == "Endzone":
                endzone_img = image.float()
            else:
                sideline_img = image.float()

        features = torch.concat(features)
        return endzone_img,sideline_img ,features, torch.tensor(label)

    def __len__(self):
        return len(self.labels)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ft_extractor = torchvision.models.resnet18(pretrained=True)
        for param in self.ft_extractor.parameters():
            param.requires_grad = False
        num_ftrs = self.ft_extractor.fc.in_features
        self.ft_extractor.fc = Identity()
        self.mlp = nn.Sequential(
            nn.Linear(22, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

        )
        self.fc = nn.Linear(64+512*2, 1)

    def forward(self, endzone_img, sideline_img, feature):

        endzone_img = self.ft_extractor(endzone_img)
        sideline_img = self.ft_extractor(sideline_img)

        feature = self.mlp(feature)
        y = self.fc(torch.cat([endzone_img,sideline_img, feature], dim=1))
        return y

def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, help="Path to job data")
    parser.add_argument("--output_folder", type=str,default="output_folder", help="Path of prediction ouput folder, default to local folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_examples", type=int, default=1000)

    # parse args
    args = parser.parse_args()

    # return args
    return args
def prepare_data(data_folder,batch_size, max_examples=None):
    frame_files = os.listdir(f"{data_folder}/extracted-train-frames/content/work/frames/train/")
    labels = pd.read_csv(f"{data_folder}/train_labels.csv")
    if max_examples:
        print("original dataset is ", len(labels), " limit number of example to ", max_examples)

        labels = labels.sample(max_examples)
    labels["frame"] =labels["step"]/10*59.95+5*59.95  
    labels["frame"] = labels["frame"].astype(int) 

    player_tracking =pd.read_csv(f"{data_folder}/train_player_tracking.csv")
    player_tracking["frame"] =player_tracking["step"]/10*59.95+5*59.95  
    player_tracking["frame"] = player_tracking["frame"].astype(int) 
    endzone_frame_files = pd.DataFrame({"endzone_img":[file for file in frame_files if "Endzone" in file]})
    sideline_frame_files = pd.DataFrame({"sideline_img":[file for file in frame_files if "Sideline" in file]})
    labels["endzone_img"] = labels.apply(lambda row: row['game_play']+"_"  +"Endzone.mp4_{:04d}.jpg".format(row['frame']),axis=1)
    labels["sideline_img"] = labels.apply(lambda row: row['game_play']+"_"  +"Sideline.mp4_{:04d}.jpg".format(row['frame']),axis=1)
    labels = labels.merge(endzone_frame_files).merge(sideline_frame_files)
    train_labels, test_labels = train_test_split(labels,test_size=0.2,stratify=labels["contact"])
    helmets = pd.read_csv(f"{data_folder}/train_baseline_helmets.csv")
    img_path = f"{data_folder}/extracted-train-frames/content/work/frames/train/"
    train_ds = NFLDataset(train_labels,helmets, player_tracking, None,img_path)
    train_dataloader = DataLoader(train_ds,batch_size=batch_size, shuffle=True)
    test_ds = NFLDataset(test_labels,helmets, player_tracking, None,img_path)
    test_dataloader = DataLoader(test_ds,batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader
def train_loop(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    mlflow.log_metric("dataset_size", size)

    for batch, (endzone_imges, sideline_imges, ft_batch, labels) in enumerate(dataloader):
        # Compute prediction and loss
        endzone_imges, sideline_imges, ft_batch, labels =endzone_imges.to(device), sideline_imges.to(device), ft_batch.to(device), labels.to(device) 
        pred = model(endzone_imges, sideline_imges, ft_batch)
        loss = loss_fn(pred.squeeze(), labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(endzone_imges)
            mlflow.log_metric("batch", batch)
            mlflow.log_metric("train_loss", loss)
            mlflow.log_metric("progress", 100*current/size)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    y_pred= []
    y_true = []

    with torch.no_grad():
        for batch, (endzone_imges, sideline_imges, ft_batch, labels) in enumerate(dataloader):
        # Compute prediction and loss
            endzone_imges, sideline_imges, ft_batch, labels =endzone_imges.to(device), sideline_imges.to(device), ft_batch.to(device), labels.to(device) 
            pred = model(endzone_imges, sideline_imges, ft_batch)
            test_loss += loss_fn(pred.squeeze(), labels).item()
            y_pred += list(torch.round(torch.sigmoid(pred).squeeze()).cpu().numpy())
            y_true += list(labels.cpu().numpy())
    test_loss /= num_batches
    mlflow.log_metric("test_loss_avg", test_loss)
    mlflow.log_metric("test_accuracy", accuracy_score(y_true,y_pred))
    mlflow.log_metric("test_f1", f1_score(y_true,y_pred))
    mlflow.log_metric("test_recall", recall_score(y_true,y_pred))

def train(train_dataloader, test_dataloader, learning_rate,epochs):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    model = Model().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        mlflow.log_metric("epoch", t)

        train_loop(train_dataloader, model, loss_fn, optimizer,device)
        test_loop(test_dataloader, model, loss_fn,device)
    print("Done!")
    mlflow.pytorch.log_model(model, "nfl_contact_detection")


def main(args):
    
    # read in data
    train_dataloader, test_dataloader = prepare_data(args.data_folder,args.batch_size, args.max_examples)
    train(train_dataloader, test_dataloader, args.learning_rate, args.epochs)

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    # run main function
    main(args)
