from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np

import warnings

import time
from Models import Unet
import Utils


def main():
    config = Utils.load_json("config.json")
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        pass
    else:
        device = torch.device('cpu')

    dataset = Utils.Brain_data(config["data"]["Brain Dummy"]["data_path"])
    # show shape of images
    '''
    for img, msk in dataset:
        print(img.shape)
        print(msk.shape)
        break
    '''
    trainset, testset = random_split(dataset, [int(
        dataset.__len__()*0.9), int(dataset.__len__()-int(dataset.__len__()*0.9))])
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=config["general"]["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=config["general"]["batch_size"])
    # show 5 training iamges
    '''
    Utils.plot_img(5, train_loader, device)
    '''

    # init model
    model = Unet.Unet((3, 256, 256))
    criterion = Unet.DiceBCELoss()
    criterion_test_1 = Unet.IoU()
    criterion_test_2 = Unet.DiceScore()


    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    
    # train
    epochs = config["general"]["epochs"]
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()

        running_train_loss = []
        running_train_IoU = []
        running_train_DiceScore = []

        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)

            pred_mask = model.forward(image)  # forward propogation
            loss = criterion(pred_mask, mask)
            IoU = criterion_test_1(pred_mask, mask)
            Dice = criterion_test_2(pred_mask,mask)
            optimizer.zero_grad()  # setting gradient to zero
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
            running_train_IoU.append(IoU.item())
            running_train_DiceScore.append(Dice.item())
        else:
            running_val_loss = []
            running_val_IoU = []
            running_val_DiceScore = []
            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    loss = criterion(pred_mask, mask)
                    IoU = criterion_test_1(pred_mask, mask)
                    Dice = criterion_test_2(pred_mask,mask)
                    running_val_loss.append(loss.item())
                    running_val_IoU.append(IoU.item())
                    running_val_DiceScore.append(Dice.item())

        epoch_train_loss = np.mean(running_train_loss)
        epoch_train_IoU = np.mean(running_train_IoU)
        epoch_train_DiceScore = np.mean(running_train_DiceScore)
        print('Train loss: {}'.format(epoch_train_loss))
        print('Train IoU: {}'.format(epoch_train_IoU))
        print('Train DiceScore: {}'.format(epoch_train_DiceScore))
        train_loss.append(epoch_train_loss)

        epoch_val_loss = np.mean(running_val_loss)
        epoch_val_IoU = np.mean(running_val_IoU)
        epoch_val_DiceScore = np.mean(running_val_DiceScore)
        print('Validation loss: {}'.format(epoch_val_loss))
        print('Validation IoU: {}'.format(epoch_val_IoU))
        print('Validation DiceScore: {}'.format(epoch_val_DiceScore))
        val_loss.append(epoch_val_loss)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
