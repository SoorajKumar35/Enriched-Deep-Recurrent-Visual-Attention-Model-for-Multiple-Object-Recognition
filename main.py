import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from network import EDRAM
from data_utils import load_data, prepare_datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.nn.utils import clip_grad_norm_
from numpytovideo import record
import cv2
import sys
from torchviz import make_dot


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def run_epoch(net, loader, optimizer=None, training=True, best_acc=0.0):

    total_loss = 0.
    total_Ly_loss = 0.
    total_LA_loss = 0.
    
    my_predictions = []
    gt_classifications = []
    net.train()
    
    if training:
        net.train()
    else:
        net.eval()
        net.apply(EDRAM.disable_running_stats)
    
    for i, (images, target) in enumerate(loader):
        #print("Current iter: ", i)
        images, target = images.to(device), target.to(device)
        
        t1 = time.time()
        locations, classifications = net(images, iteration=i)
        probs = net.prob_layer(classifications)
        #print("net time: %f" % (time.time() - t1))
        
        predicted_classes = torch.max(probs, dim=2)
        gt_classes = target[:,6:]
        #gt_classes = gt_classes.nonzero()[:,1]
        
        
        my_predictions += list(predicted_classes[1][:,-1].cpu().data.numpy())
        gt_classifications += list(gt_classes.cpu().data.numpy()[:,0])

        Ly_loss, LA_loss = net.loss(locations, classifications, target)
        total_loss = total_loss + Ly_loss + LA_loss
        total_Ly_loss += Ly_loss
        total_LA_loss += LA_loss
        
        loss = Ly_loss + LA_loss
        
        
        if training:
            
            t2 = time.time()
            optimizer.zero_grad()
            loss.backward()
            #process_gradients(net, 20.0)
            clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            #print("backprop time: %f" % (time.time() - t2))
            
            if (i+1) % 10 == 0:
                #print(target[0])
                print("Iter [%d/%d] Average loss: %.4f. Average Ly_loss: %.4f. Average LA_loss: %.4f." % (i+1, len(loader), total_loss / (i+1), total_Ly_loss / (i+1), total_LA_loss / (i+1)))
               
        
                
    print("Average loss for epoch: %f. Average Ly_loss: %f. Average LA_loss: %f." % (total_loss / (len(loader)), total_Ly_loss / len(loader), total_LA_loss / len(loader)))
    
    my_predictions = np.array(my_predictions)
    gt_classifications = np.array(gt_classifications)
    pred_acc = np.sum(my_predictions == gt_classifications) / len(my_predictions)
    print("Prediction accuracy: %f" % (np.sum(my_predictions == gt_classifications) / len(my_predictions)))

    #plot_confusion_matrix(my_predictions, gt_classifications)
    net.apply(EDRAM.enable_running_stats)
    if training:
        torch.save(net.state_dict(), 'network.pth')
    """
    if not training and pred_acc > best_acc:
        best_acc = pred_acc
        torch.save(net.state_dict(), 'best_network.pth')
    """

def process_gradients(network, clip):
    #grad_sum = 0
    for param in network.parameters():
        if param.grad is None:
            continue
        #print(param.grad.data.size())
        """
        if (param.grad.data.size() == torch.Size([6])):
            print(param.grad.data)
        """
        param.grad.data = param.grad.data.clamp(-clip, clip)   
        #grad_sum += torch.sum( param.grad.data ** 2 )
    #print(grad_sum)


"""
    Plot confusion matrix to see how good each class predictor is doing
"""


def plot_confusion_matrix(predictions, gt):
    # confuson matrix code from cs 440, should work, haven't tested
    print(gt, predictions)
    matrix = confusion_matrix(gt, predictions)
    classes = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    print("Confusion matrix:")
    print(matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title="Confusion Matrix",
           ylabel="Actual",
           xlabel="Predicted")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i,j], ".2f"),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    plt.pause(0.001)
    
    
def grid_generator(h,w):
        """
        This function is used to create the meshgrid coords to be used
        :return:
        """

        x_t, y_t = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
        meshgrid = np.concatenate((np.matrix(x_t.flatten()), np.matrix(y_t.flatten()), np.ones((1, h*w))),
                                        axis=0)
        return meshgrid
    
def generate_video(net, loader):

    frames = []
    captions = []

    for i, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        
        t1 = time.time()
        locations, classifications = net(images, iteration=i)
        
        
        predicted_classes = torch.max(classifications, dim=2)
        gt_classes = target[:,6:]
        #gt_classes = gt_classes.nonzero()[:,1]
        
        
        my_predictions = list(predicted_classes[1][:,-1].cpu().data.numpy())
        
        for j in range(8):
        
            im = images[j,0].cpu().data.numpy()
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            label = target.cpu().data.numpy()[j,6]
            
            for k in range(locations.shape[1]):
                
                """
                if (j == 0 and k == 0):
                    g1 = make_dot(locations[0])
                    g2 = make_dot(classifications[0])
                    g1.render('location.gv', view=True)
                    g2.render('classifications.gv', view=True)
                """ 
            
                location = locations[j,k].cpu().data.numpy().reshape((2,3))
                print(location)
                myh, myw = 120, 120
                
                grid = grid_generator(myh, myw)
                
                coords = (100 * np.matmul(location, grid)).astype(np.int32).T
                coords = np.clip(coords, 0, 99)
                
                background = im / 3
                background[coords[:,0],coords[:,1]] = im[coords[:,0],coords[:,1]]
                
                background = (background * 255.).astype(np.uint8).reshape((100,100,1))
                background = np.repeat(background, 3, axis=2)
                
                red_coords = np.concatenate([coords[:myh], coords[:-myw:myw], coords[-myh:], coords[myw-1::myw]], axis=0)

                #print(red_coords)
                for l in range(-1,2):
                    for m in range(-1,2):
                        x = np.clip(red_coords[:,1]+l,0,99)
                        y = np.clip(red_coords[:,0]+l,0,99)
                        background[y,x,2] = 255
                                
                frames.append(background)
                captions.append("Prediction: %d. Correct Label: %d." % (int(my_predictions[j]), int(label)))
                cv2.imwrite("glimpse"+str(j)+str(k)+".jpg", background)
            
    
        if i == 0:
            break
    
    record(np.array(frames), "EDRAM.avi", 1)

def main():

    training = False

    num_epochs = 100
    best_acc = 0.0
    # adjust this as necessary
    train_batch_size, val_batch_size, test_batch_size = 128, 128, 128

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data('.')
    # Define data loaders
    train_loader, test_loader, val_loader = prepare_datasets(
        X_train, y_train, X_valid, y_valid, X_test, y_test, train_batch_size,
        val_batch_size, test_batch_size)

    # Define hyperparameter dict - Follows values from MNIST cluttered net in paper
    MNIST_cluttered_config = {
        "A_size": 6,
        "T": 6,
        "n_classes": 10,
        "context_downsample": (12, 12),
        "glimpse_size": (26, 26),
        "hidden1": 512,
        "hidden2": 512,
        "context_filters": [[5, 3, 3], [1, 16, 16, 32]],
        "glimpse_filters": [[3, 3, 3, 3, 3, 3, 3], [1, 64, 64, 128, 128, 160, 192]],
        "glimpse_fcs": [1024],
        "classification_fcs": [512, 1024, 1024, 10],
        "emission_fcs": [512, 6],
        "LA_weights": [1, 0.5, 1, 0.5, 1, 1],
        "loss_term_weights": [1, 1]
    }
    
    # Init network
    net = EDRAM(MNIST_cluttered_config).to(device)
    print(net)
    
    
    if not training:
        #LOADS BEST NET
        net.load_state_dict(torch.load('best_network.pth'))
        print("TESTING BEST MODEL")
        if (not training):
            with torch.no_grad():
                run_epoch(net, test_loader, training=False)
                run_epoch(net, train_loader, training=False)
            #generate_video(net, test_loader)
        sys.exit()

    
    #net.load_state_dict(torch.load('network.pth'))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, eps=1e-5, momentum=0.9)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.9, momentum=0.8)
   
    lambda1 = lambda epoch: 0.96 ** epoch
    lambda2 = lambda epoch: 1 / (10 ** min(2, int(epoch / 15)))
    lambda3 = lambda epoch: 1 / (10 ** min(1, int(epoch / 5)))
    lambda4 = lambda epoch: max((0.794 ** min(10, int(epoch / 10))), 0.1)
    lambda5 = lambda epoch: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda1 )
    # Training loop
    for epoch in range(1, num_epochs):
        print("\n   ----------------   \n\nSTARTING TRAINING ITERATION %d\n\n   ----------------   \n\n" % (epoch))
        with torch.no_grad():
            run_epoch(net, val_loader, training=False, best_acc=best_acc)
        run_epoch(net, train_loader, optimizer=optimizer)
        scheduler.step()
        for g in optimizer.param_groups:
            print(g['lr'])
        # Run model over validation dataset
        with torch.no_grad():
            run_epoch(net, val_loader, training=False, best_acc=best_acc)
        if (epoch % 5 == 0):
            with torch.no_grad():
                run_epoch(net, test_loader, training=False)

    # Testing loop
    with torch.no_grad():
        run_epoch(net, test_loader, training=False)


if __name__ == "__main__":
    main()
