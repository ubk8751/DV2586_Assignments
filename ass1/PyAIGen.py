# License: BSD
# Author: Sasank Chilamkurthy
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import copy

from constants import constant

c = constant('./ass1/MiniDIDA.ds')
cudnn.benchmark = True
plt.ion()   # interactive mode
criterion = nn.CrossEntropyLoss()
inputs, classes = next(iter(c.dataloader['train']))
out = torchvision.utils.make_grid(inputs)

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def get_confusion_matrix(output, nb_classes:int=10):
    nb_samples = 400
    output = torch.randn(nb_samples, nb_classes)
    pred = torch.argmax(output, 1)
    target = torch.randint(0, nb_classes, (nb_samples,))

    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(target, pred):
        conf_matrix[t, p] += 1

    print('Confusion matrix\n', conf_matrix)

    TP = conf_matrix.diag()
    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()
        
        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
    return conf_matrix, TP, TN, FP, FN

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in c.dataloader[phase]:
                inputs = inputs.to(c.device)
                labels = labels.to(c.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / c.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / c.dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    output = model(inputs)
    conf_matrix, TP, TN, FP, FN = get_confusion_matrix(output=output, nb_classes=10)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, conf_matrix, TP, TN, FP, FN

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(c.dataloader['val']):
            inputs = inputs.to(c.device)
            labels = labels.to(c.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {c.class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def calc_f1(TP, TN, FP, FN):
    true_TP = 0
    for item in TP.tolist():
        true_TP += item
    FP = FP.item()
    FN = FN.item()
    TN = TN.item()
    if true_TP == 0.0 or FP == 0.0 or FN == 0:    
        if true_TP == 0.0:
            print("Model has 0 true positives")
        if FP == 0.0:
            print("Model has 0 false positives")
        if FN == 0.0:
            print("Model has 0 false negatives")
        if (true_TP == 0.0 and FP == 0.0) or (true_TP == 0 and FN == 0):
            return 0
    precision = true_TP/(true_TP+FP)
    recall = true_TP/(true_TP+FN)
    return 2*((precision*recall)/(precision+recall))

def vgg19(num_epochs:int=25, lr:float=0.001, momentum:float=0.9, step_size:int=7, gamma:float=0.1, v_model:bool=False):
    vgg = models.vgg19(weights = None)
    num_ftrs = vgg.fc.in_features
    out_sample = len(c.class_names)
    vgg.fc = nn.Linear(num_ftrs, out_sample)
    vgg = vgg.to(c.device)
    optimizer_ft = optim.SGD(vgg.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    vgg, conf_matrix, TP, TN, FP, FN = train_model(vgg, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    F1 = calc_f1(TP, TN, FP, FN)
    if v_model:
        visualize_model(vgg)
    return vgg, conf_matrix, TP, TN, FP, FN, F1

def ResNet50(num_epochs:int=25, lr:float=0.001, momentum:float=0.9, step_size:int=7, gamma:float=0.1, v_model:bool=False):
    RN50 = models.resnet50(weights = None)
    num_ftrs = RN50.fc.in_features
    out_sample = len(c.class_names)
    RN50.fc = nn.Linear(num_ftrs, out_sample)
    RN50 = RN50.to(c.device)
    optimizer_ft = optim.SGD(RN50.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    RN50, conf_matrix, TP, TN, FP, FN = train_model(RN50, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    F1 = calc_f1(TP, TN, FP, FN)
    if v_model:
        visualize_model(RN50)
    return RN50, conf_matrix, TP, TN, FP, FN, F1

def DenseNet(num_epochs:int=25, lr:float=0.001, momentum:float=0.9, step_size:int=7, gamma:float=0.1, v_model:bool=False):
    DN = models.DenseNet(weights = None)
    num_ftrs = DN.fc.in_features
    out_sample = len(c.class_names)
    DN.fc = nn.Linear(num_ftrs, out_sample)
    DN = DN.to(c.device)
    optimizer_ft = optim.SGD(DN.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    DN, conf_matrix, TP, TN, FP, FN = train_model(DN, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    F1 = calc_f1(TP, TN, FP, FN)
    if v_model:    
        visualize_model(DN)
    return DN, conf_matrix, TP, TN, FP, FN, F1

# Main function to test module
def main():
    model_ft = models.resnet50(weights=None)
    num_ftrs = model_ft.fc.in_features
    out_sample = len(c.class_names)
    model_ft.fc = nn.Linear(num_ftrs, out_sample)
    model_ft = model_ft.to(c.device) 
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft, conf_matrix, TP, TN, FP, FN, F1 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=1)
    F1 = calc_f1(TP, TN, FP, FN)
    print("FF = " + str(F1))
    visualize_model(model_ft)
    
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()