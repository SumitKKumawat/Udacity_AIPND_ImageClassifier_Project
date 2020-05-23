#import module normalize.py (common for train.py and predict.py)
from normalize import *

# getting user inputs
#default parameters like model vgg, learning rate 0.001, epochs 5
parser = argparse.ArgumentParser(description="Trains a network on a dataset of images and saves the model to a checkpoint")
parser.add_argument('data_dir', type=str, help='set the data directory')
parser.add_argument('--arch', default = 'vgg13', type=str, help='choose the model architecture')
parser.add_argument('--learning_rate', default=0.001, type=float, help='the learning rate')
parser.add_argument('--hidden_units', default=25088, type=int, help='the sizes of the hidden layers')
parser.add_argument('--epochs', default=5, type=int, help='number of training epochs')
parser.add_argument('--gpu', help='set the gpu mode')
parser.add_argument('--save_dir', default = '', type=str, help='set the checkpoint path')
args = parser.parse_args()


data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu

# if user doen't provide inputs, set default (already there in parser as well)
if (arch == "vgg13"):
    input_size = 25088
    output_size = 102
elif (arch == "densenet121"):
    input_size = 1024
    output_size = 102
else:
    print("Please select model architectures vgg13 or densenet121.")
    exit()

#when save_dir is not provided by user
if save_dir is None:
    save_dir = "save_checkpoint.pth"

#when learning_rate is not provided by user    
if learning_rate is None:
    learning_rate = 0.001
else:
    learning_rate = float(learning_rate)

#when hidden_units is not provided by user
if hidden_units is None:
    if (arch == "vgg13"):
        hidden_units = 4096
    elif (arch == "densenet121"):
        hidden_units = 500
else:
    hidden_units = int(hidden_units)

#when epochs is not provided by user    
if epochs is None:
    epochs = 10
else:
    epochs = int(epochs)

#when device is not provided by user    
if device is None:
    device = "cpu"

#if user inputs for everything is none then exit, required inputs from user.
if(data_dir == None) or (save_dir == None) or (arch == None) or (learning_rate == None) or (hidden_units == None) or (epochs == None) or (device == None):
    print("data_dir, arch , learning_rate, hidden_units, and epochs cannot be none, please provide inputs")
    exit()


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {d: torch.utils.data.DataLoader(image_datasets[d], batch_size=32, shuffle=True) for d in ['train', 'valid', 'test']} 

# TODO: Build and train your network with vgg13 or densenet121
if (arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
elif (arch == 'densenet121'):
    model = models.densenet121(pretrained=True)
model

# TODO: Do validation on the test set
# now freezing the parameters hence backpropogation not happen
for param in model.parameters():
    param.requires_grad = False

# building the feed forward network - vgg unidirectional model
#taking relu as activation function with dropout and using fully connected layers.
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                        ('fc2', nn.Linear(hidden_units, output_size)),
                                        ('output', nn.LogSoftmax(dim=1))]))

# applying the classifier on vgg network model
model.classifier = classifier


# training the model with pretrained vgg network
#setting up lerning rate to 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)
print("Model Learning Starts..... (in progress)")
for e in range(epochs):

    for dataset in ['train', 'valid']:
        if dataset == 'train':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(dataset == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if dataset == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        epoch_loss = running_loss / dataset_sizes[dataset]
        epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
        
        print("Epoch: {}/{}... ".format(e+1, epochs),
              "{} Loss: {:.4f}    Accurancy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))
        
# checking validation accuracy on test dataset
def check_accuracy_on_test(test_loader):    
    correct = 0
    total = 0
    #model.to('cuda:0')
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#printing network accuracy once model run completes            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(dataloaders['train'])

# TODO: Save the checkpoint 
#saving the model ex. vgg (weights will be saved)
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'model': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
print("Save model to:" + save_dir)