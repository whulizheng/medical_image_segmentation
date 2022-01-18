from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import albumentations as A
import warnings
import time
import Utils


def device_prepare():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device('cuda:0')

    else:
        print("Using CPU")
        return torch.device('cpu')


def define_augmentation(if_aug, width, height):
    if if_aug:
        return A.Compose([
            A.Resize(width=width, height=height, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04,
                               rotate_limit=0, p=0.25),
        ])
    else:
        return A.Compose([
            A.Resize(width=width, height=height, p=1.0),
        ])


def dataset_prepare(config, transform):
    dataset = Utils.Brain_data(
        config["data"][config["general"]["datasets"][config["general"]["chosen_dataset"]]]["data_path"], transform)
    trainset, testset = random_split(dataset, [int(
        dataset.__len__()*0.9), int(dataset.__len__()-int(dataset.__len__()*0.9))])
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=config["general"]["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=config["general"]["batch_size"])
    return train_loader, test_loader


def init_evaluation(name, config):
    if name == "DiceScore":
        from Evaluations import DiceScore
        method = DiceScore.DiceScore()
        return method
    elif name == "IoU":
        from Evaluations import IoU
        method = IoU.IoU()
        return method
    else:
        print("Wrong Evaluation Name")
        exit(-1)


def init_models(name, config):
    if name == "Unet":
        from Models import Unet
        model = Unet.Model((config["general"]["input_channels"],
                           config["general"]["width"], config["general"]["height"]))
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = Unet.Loss()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "Unet_pro":
        from Models import Unet_pro
        model = Unet_pro.Model(
            (config["general"]["input_channels"], config["general"]["width"], config["general"]["height"]))
        criterion = Unet_pro.Loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "Transformer_CNN_Unet_mix_p1":
        from Models import Transformer_CNN_Unet_mix_p1
        model = Transformer_CNN_Unet_mix_p1.Model(
            (config["general"]["input_channels"], config["general"]["width"], config["general"]["height"]))
        criterion = Transformer_CNN_Unet_mix_p1.Loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "Transformer_CNN_Unet_mix_p2":
        from Models import Transformer_CNN_Unet_mix_p2
        model = Transformer_CNN_Unet_mix_p2.Model(
            (config["general"]["input_channels"], config["general"]["width"], config["general"]["height"]))
        criterion = Transformer_CNN_Unet_mix_p2.Loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "Transformer_CNN_Unet_mix_p3":
        from Models import Transformer_CNN_Unet_mix_p3
        model = Transformer_CNN_Unet_mix_p3.Model(
            (config["general"]["input_channels"], config["general"]["width"], config["general"]["height"]))
        criterion = Transformer_CNN_Unet_mix_p3.Loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "Transformer_pure":
        from Models import Transformer_pure
        model = Transformer_pure.Model(
            (config["general"]["input_channels"], config["general"]["width"], config["general"]["height"]))
        criterion = Transformer_pure.Loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    else:
        print("Wrong Model Name")
        exit(-1)


def train(model,criterion,optimizer, device, train_loader, test_loader, epochs):
    train_loss = []
    test_loss = []
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
            running_test_loss = []
            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    loss = criterion(pred_mask, mask)
                    running_test_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        print('Train loss: {}'.format(epoch_train_loss))
        train_loss.append(epoch_train_loss)

        epoch_test_loss = np.mean(running_test_loss)
        print('Test loss: {}'.format(epoch_test_loss))
        test_loss.append(epoch_test_loss)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return model, train_loss, test_loss


def main():
    config = Utils.load_json("config.json")
    device = device_prepare()
    Utils.show_config(config)
    # augmentation
    transform = define_augmentation(
        config["general"]["augmentation"], config["general"]["width"], config["general"]["height"])
    if config["general"]["augmentation"]:
        print("Data Augmenting...")
    # prepare dataset
    train_loader, test_loader = dataset_prepare(config, transform)
    # train model
    for i in config["general"]["chosen_models"]:
        # init model
        model_name = config["general"]["models"][i]
        print("Training model: "+model_name)
        model, criterion, optimizer = init_models(model_name, config)
        model.to(device)

        # train
        epochs = config["general"]["epochs"]
        model, train_loss, test_loss = train(
            model,criterion,optimizer, device, train_loader, test_loader, epochs)

        # evaluation
        evaluations = {}
        for e in config["evaluation"]["chosen_methods"]:
            method_name = config["evaluation"]["methods"][e]
            print("Evaluating Model: "+model_name+" By "+method_name)
            method = init_evaluation(method_name, config)
            scores = []
            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    score = method(pred_mask, mask)
                    scores.append(score.item())
            evaluations[method_name] = np.mean(scores)
            print("Model: "+model_name+" Got "+method_name +
                  ": "+str(float(np.mean(scores))))

        # save_log
        Utils.save_log(model_name, config, train_loss, test_loss, evaluations=evaluations)

        # save_model
        if config["general"]["if_save_models"]:
            date = time.strftime('%Y_%m_%d_%H_%M_%S',
                                 time.localtime(time.time()))
            path = "Saved_models/" + model_name + "_" + date + ".pt"
            torch.save(model, path)
            print("model "+model_name+" saved at: "+path)
        
        del model, criterion, optimizer


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
