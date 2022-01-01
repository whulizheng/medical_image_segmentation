from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np

import warnings

import time
from Models import Unet2
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
    model = Unet2.UNet((3, 256, 256))
    criterion = Unet2.DiceLoss()
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

        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)

            pred_mask = model.forward(image)  # forward propogation
            loss = criterion(pred_mask, mask)
            optimizer.zero_grad()  # setting gradient to zero
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
        else:
            running_val_loss = []
            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    loss = criterion(pred_mask, mask)
                    running_val_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        print('Train loss: {}'.format(epoch_train_loss))
        train_loss.append(epoch_train_loss)

        epoch_val_loss = np.mean(running_val_loss)
        print('Validation loss: {}'.format(epoch_val_loss))
        val_loss.append(epoch_val_loss)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
