import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    # Run any number of epochs you want
    ep = 10

    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []


    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt, ave_loss = 0, 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model

            out = model(x)
            # Calculate loss
            print(out.shape, label.shape)
            loss = criterion(out, label)

            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        training_accuracy.append(correct_cnt/total_cnt)
        training_loss.append(ave_loss)

        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        val_loss, val_acc = 0, 0
        correct_cnt, total_loss, total_cnt, ave_loss = 0, 0, 0, 0

        for batch, (x, label) in enumerate(val_loader,1):
            pred_val = model(x)
            loss = criterion(pred_val, label)

            val_loss += loss.item()
            _, pred_label = torch.max(pred_val, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            if batch == len(val_loader):
                val_loss = val_loss / batch
                val_acc = correct_cnt / total_cnt
                print('Validation batch index: {}, val_loss: {:.6f}, acc" {:.3f}'.format(
                    batch, val_loss, val_acc))

        validation_accuracy.append(val_acc)
        validation_loss.append(val_loss)




        model.train()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    print(model)

    # Plot Learning Curve
    # TODO
    epoch = list(range(ep))

    fig1 = plt.figure()
    plt.plot(epoch, training_accuracy, 'r')
    plt.plot(epoch, validation_accuracy, 'b')
    plt.gca().legend(('training_accuracy', 'validation_accuracy'), loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accracy-Epoch')
    plt.show(fig1)


    fig2 = plt.figure()
    plt.plot(epoch, training_loss, 'r')
    plt.plot(epoch, validation_loss, 'b')
    plt.gca().legend(('training_loss', 'validation_loss'), loc = 'best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss-Epoch')
    plt.show(fig2)



  

