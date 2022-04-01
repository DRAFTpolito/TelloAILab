import torch
import torch.optim as optim
from torch import nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import cv2
import numpy as np
import os


class Trainer:
    def __init__(self, data_folder='data', num_classes=4, starting_lr=0.0001, batch_size=8, saved_model=None):
        self.img_scaling = 1/2

        # Define the neural network
        self.model = models.resnet34(pretrained=True)
        # freeze all the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = True

        num_ftrs = self.model.fc.in_features

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=-1)
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if saved_model and os.path.isfile(saved_model):
            self.model.load_state_dict(torch.load(saved_model, map_location=torch.device(self.device)))
        else:
            print("Model path is not a file.")
        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=starting_lr, amsgrad=True)
        # Learning Rate schedule
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        if not os.path.isdir(data_folder):
            print("Data folder not found, dataset not loaded and dataloaders not created.")
            return

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3),
                transforms.GaussianBlur((3, 3), sigma=(1.0, 2.0)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder, x),
                                                  data_transforms[x]) for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=4) for x in ['train', 'val']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes

    # Train one epoch
    def train(self, log_interval, epoch):
        """Trains a neural network for one epoch.
        Args:
            log_interval: the log interval.
            epoch: the number of the current epoch.
        Returns:
            the cross entropy Loss value on the training data.
            the accuracy on the training data.
        """
        correct = 0
        samples_train = 0
        loss_train = 0
        size_ds_train = len(self.dataloaders['train'].dataset)
        num_batches = len(self.dataloaders['train'])

        self.model.train(True)
        for idx_batch, (images, labels) in enumerate(self.dataloaders['train']):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            scores = self.model(images)

            loss = self.criterion(scores, labels)
            loss_train += loss.item() * len(images)
            samples_train += len(images)

            loss.backward()
            self.optimizer.step()
            correct += get_correct_samples(scores, labels)

            if log_interval > 0:
                if idx_batch % log_interval == 0:
                    running_loss = loss_train / samples_train
                    global_step = idx_batch + (epoch * num_batches)

        loss_train /= samples_train
        accuracy_training = 100. * correct / samples_train
        return loss_train, accuracy_training

    def validate(self):
        """Evaluates the model.
        Args:
        Returns:
            the loss value on the validation data.
            the accuracy on the validation data.
        """
        correct = 0
        samples_val = 0
        loss_val = 0.
        self.model.eval()
        with torch.no_grad():
            for idx_batch, (images, labels) in enumerate(self.dataloaders['val']):
                images, labels = images.to(self.device), labels.to(self.device)
                scores = self.model(images)

                loss = self.criterion(scores, labels)
                loss_val += loss.item() * len(images)
                samples_val += len(images)
                correct += get_correct_samples(scores, labels)

        loss_val /= samples_val
        accuracy = 100. * correct / samples_val
        return loss_val, accuracy

    def training_loop(self, epochs, log_interval, verbose):
        """Executes the training loop.
            Args:
                epochs: the number of epochs.
                log_interval: intervall to print on tensorboard.
                verbose: if true print the value of loss.
            Returns:
                A dictionary with the statistics computed during the train:
                the values for the train loss for each epoch.
                the values for the train accuracy for each epoch.
                the values for the validation accuracy for each epoch.
        """
        losses_values = []
        train_acc_values = []
        val_acc_values = []
        for epoch in range(1, epochs + 1):
            loss_train, accuracy_train = self.train(log_interval, epoch)
            loss_val, accuracy_val = self.validate()

            losses_values.append(loss_train)
            train_acc_values.append(accuracy_train)
            val_acc_values.append(accuracy_val)

            lr = self.optimizer.param_groups[0]['lr']

            if verbose:
                print(f'Epoch: {epoch} '
                      f' Lr: {lr:.8f} '
                      f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                      f' Accuracy: Train = [{accuracy_train:.2f}%] - Val = [{accuracy_val:.2f}%] ')

            # Increases the internal counter
            if self.scheduler:
                self.scheduler.step()

        return {'loss_values': losses_values,
                'train_acc_values': train_acc_values,
                'val_acc_values': val_acc_values}

    def execute(self, epochs, name_train="hands_trainer"):
        """Executes the training loop.
        Args:
            name_train: the name for the log subfolder.
            epochs: the number of epochs.
        """
        # Visualization
        log_interval = 20
        log_dir = os.path.join("logs", name_train)

        statistics = self.training_loop(epochs, log_interval, verbose=True)

        best_epoch = np.argmax(statistics['val_acc_values']) + 1
        best_accuracy = statistics['val_acc_values'][best_epoch - 1]

        print(f'Best val accuracy: {best_accuracy:.2f} epoch: {best_epoch}.')

    def inference(self, img):
        # Resize frame of video for faster processing
        small_frame = cv2.resize(img, (0, 0), fx=self.img_scaling, fy=self.img_scaling)
        process_frame = self.preprocess(small_frame)
        return self.model(process_frame)

    def preprocess(self, img):
        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (224, 224))
        x = cv2.medianBlur(x, 9)  # Reduce impulse noise
        x = cv2.GaussianBlur(x, (3, 3), 1.5)  # Reduce linear noise
        x = torch.from_numpy(x).float()
        x = x.to(self.device)
        x = x[None, None, ...]  # Adding batch and channel dimensions
        return x

    def save_model(self, model_path):
        torch.save(self.model.get_state_dict(), model_path)


# Accuracy
def get_correct_samples(scores, labels):
    """Gets the number of correctly classified examples.
    Args:
        scores: the scores predicted with the network.
        labels: the class labels.
    Returns:
        the number of correct samples.
    """
    classes_predicted = torch.argmax(scores, 1)
    return (classes_predicted == labels).sum().item()
