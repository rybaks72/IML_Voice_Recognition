import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

#from src.mobilenet_model import MobileNetV2
from src.model import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from datetime import datetime
from preprocessing_pipeline.util import random_data_split
from src.resnet_model import ResNet18
# from src.googlenet_model import GoogleNet
# from src.mobilenet_model import MobileNetV2

def train():
    #model, inputs and labels have to be on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # changed path here
    #IMPORTANT run it from IML_voice_recognition directory using python -m src.train or at least this way it works for me
    # if you want to run it from the src directory change the path below to "../preprocessing_pipeline" but in my case the imports did not work AND THEN
    # you have to change the path to models, results and tensor_board_outputs directory
    # I will try to fix that later so that you can change it in 1 place or maybe nowhere
    data = random_data_split(path="./preprocessing_pipeline")
    x_train, y_train = data['train']
    x_valid, y_valid = data['validate']
    x_test, y_test = data['test']

    x_train = x_train.float()
    x_valid = x_valid.float()
    x_test = x_test.float()
    batch_size =32
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=8, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=8, shuffle=False)
    weights = torch.tensor([1.0, data['weight']]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    #model_name = "first_trial"
    #net = SimpleCNN().to(device)

    model_name = "resnet_trial_222_32_input"
    net = ResNet18().to(device)

    # model_name = "googlenet_trial"
    # net = GoogleNet().to(device)

    # model_name = "mobilenet_trial"
    # net = MobileNetV2().to(device)

    #criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model_loss = float('inf')

    #directory for results csv file
    results_directory = "./results"
    os.makedirs(results_directory, exist_ok=True)
    filename_results = "./results/tests_results.csv"
    # id is assigned automatically based on how many rows we have in the filename_results file
    experiment_id = get_experiment_id(filename_results)

    #directory with models
    os.makedirs("./models", exist_ok=True)

    writer = SummaryWriter(log_dir=f"./tensor_board_outputs/id_{experiment_id}_{model_name}")

    max_epochs = 15
    best_epoch = 0
    print("TRAINING START")
    for epoch in range(max_epochs):  # loop over the dataset multiple times, this should be adjusted later
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            print(f"inputs shape: {inputs.shape}")
            inputs, labels = inputs.float().to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            current_batch_num = epoch * len(train_loader) + i

            val_loss = validate(net, criterion, val_loader, device)

            # one plot with both
            writer.add_scalars('loss', {
                'train': loss.item(),
                'validation': val_loss
            }, current_batch_num)

            #save the best model yet
            if val_loss < best_model_loss:
                best_model_loss = val_loss
                best_epoch = epoch
                model_path = f"./models/id_{experiment_id}_{model_name}.pth"
                torch.save({'epoch': epoch, 'batch': i, 'batch_num': current_batch_num,
                            'net_state_dict': net.state_dict(),'val_loss': val_loss }, model_path)
                print(f"BEST (val_loss: {val_loss:.4f}) epoch:{epoch} batch:{i} batch_num:{current_batch_num} train_loss:{loss.item():.4f}")
            else:
                print(f"No improvement: val_loss: {val_loss:.4f} epoch:{epoch} batch:{i} batch_num:{current_batch_num} train_loss:{loss.item():.4f}")

            # return to training mode after validation
            net.train()

        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.3f}")


    print("Testing the best model")
    net.load_state_dict((torch.load(f"./models/id_{experiment_id}_{model_name}.pth", map_location=device))['net_state_dict'])
    test_metrics = calculate_metrics(net, test_loader, device)
    val_metrics = calculate_metrics(net, val_loader, device)
    train_metrics = calculate_metrics(net, train_loader, device)
    save_to_csv_experiment_results(filename_results,
                                   experiment_id,
                                   model_name,
                                   test_metrics,
                                   val_metrics,
                                   train_metrics,
                                   lr = learning_rate,
                                   batch_size= batch_size,
                                   max_epochs=max_epochs,
                                   best_epoch=best_epoch,
                                   notes="input 64 again to test")
                                   
    print("TEST END")

    writer.close()





def validate(net: SimpleCNN, criterion, valloader: DataLoader, device):
    net.eval()
    total_loss = 0.0
    correct_class0 = 0
    total_class0 = 0
    correct_class1 = 0
    total_class1 = 0
    with torch.no_grad():
        for inputs,labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            for label, prediction in zip(labels, predicted):
                if int(label.item()) == 0:
                    total_class0 += 1
                    if int(prediction.item()) == 0: correct_class0 += 1
                elif int(label.item()) == 1:
                    total_class1 += 1
                    if int(prediction.item()) == 1: correct_class1 += 1
        avg_loss = total_loss / len(valloader)

        acc0 = correct_class0 / total_class0 if total_class0 > 0 else 0
        acc1 = correct_class1 / total_class1 if total_class1 > 0 else 0

        # Log to TensorBoard
        print(f"\n>>> [VAL REPORT] Loss: {avg_loss:.4f} | Class 0 (Imposters): {acc0:.1f} | Class 1 (You): {acc1:.1f}")
        return avg_loss


def calculate_metrics(net: SimpleCNN, dataloader: DataLoader, device):
    all_predictions = []
    all_labels = []
    net.eval()
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _,predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #returns array [[tn, fp],[fn, tp]]
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.shape == (2,2):
        true_negative, false_positive, false_negative, true_positive = cm.ravel().tolist()
    else:
        print(f"WARNING: Model only predicting one class: {set(all_predictions)}")
        true_negative, false_positive, false_negative, true_positive = (0,0,0,0)


    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0) #tp / (tp + fp)
    recall = recall_score(all_labels, all_predictions) #tp / (tp + fn)

    total_class0 = false_positive + true_negative
    total_class1 = false_negative + true_positive

    # False Acceptance Ratio – the total number of incorrectly “accepted” instances
    # from Class 0 divided by the number of samples from Class 0.
    FAR = false_positive / total_class0 if total_class0 > 0 else 0.0

    #False Rejection Ratio – the total number of incorrectly “rejected” instances
    #from Class 1 divided by the number of samples from Class 1.
    FRR = false_negative / total_class1 if total_class1 > 0 else 0.0

    #f1 = 2*TP/(2*TP+FP+FN) and with different parameter average:
    #metrics for each class are returned
    f1_none = f1_score(all_labels, all_predictions, average=None, labels=[0, 1])

    f1_none_class0, f1_none_class1 = f1_none[0], f1_none[1]

    # Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)


    results = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1_none_class0":f1_none_class0,
               "f1_none_class1": f1_none_class1,
               "f1_macro": f1_macro,
               "true_negative": true_negative,
               "false_positive": false_positive,
               "false_negative": false_negative,
               "true_positive": true_positive,
               'FAR': FAR,
               'FRR': FRR
               }

    return results

def save_to_csv_experiment_results(
    filename,
    experiment_id,
    experiment_name,
    test_metrics,
    val_metrics,
    train_metrics,
    lr,
    batch_size,
    max_epochs,
    best_epoch,
    notes=None,
):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    header = [
        "experiment_id",
        "timestamp",
        "experiment_name",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_none_class0",
        "test_f1_none_class1",
        "test_f1_macro",
        "test_true_negative",
        "test_false_positive",
        "test_false_negative",
        "test_true_positive",
        "test_FAR",
        "test_FRR",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1_none_class0",
        "val_f1_none_class1",
        "val_f1_macro",
        "val_true_negative",
        "val_false_positive",
        "val_false_negative",
        "val_true_positive",
        "val_FAR",
        "val_FRR",
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1_none_class0",
        "train_f1_none_class1",
        "train_f1_macro",
        "train_true_negative",
        "train_false_positive",
        "train_false_negative",
        "train_true_positive",
        "train_FAR",
        "train_FRR",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "best epoch",
        "notes"
    ]

    row = [
        experiment_id,
        timestamp,
        experiment_name,
        round(test_metrics['accuracy'], 4),
        round(test_metrics['precision'], 4),
        round(test_metrics['recall'], 4),
        round(test_metrics['f1_none_class0'], 4),
        round(test_metrics['f1_none_class1'], 4),
        round(test_metrics['f1_macro'], 4),
        test_metrics['true_negative'],
        test_metrics['false_positive'],
        test_metrics['false_negative'],
        test_metrics['true_positive'],
        test_metrics['FAR'],
        test_metrics['FRR'],
        round(val_metrics['accuracy'], 4),
        round(val_metrics['precision'], 4),
        round(val_metrics['recall'], 4),
        round(val_metrics['f1_none_class0'], 4),
        round(val_metrics['f1_none_class1'], 4),
        round(val_metrics['f1_macro'], 4),
        val_metrics['true_negative'],
        val_metrics['false_positive'],
        val_metrics['false_negative'],
        val_metrics['true_positive'],
        val_metrics['FAR'],
        val_metrics['FRR'],
        round(train_metrics['accuracy'], 4),
        round(train_metrics['precision'], 4),
        round(train_metrics['recall'], 4),
        round(train_metrics['f1_none_class0'], 4),
        round(train_metrics['f1_none_class1'], 4),
        round(train_metrics['f1_macro'], 4),
        train_metrics['true_negative'],
        train_metrics['false_positive'],
        train_metrics['false_negative'],
        train_metrics['true_positive'],
        train_metrics['FAR'],
        train_metrics['FRR'],
        lr,
        batch_size,
        max_epochs,
        best_epoch,
        notes if notes is not None else ""
    ]

    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        #if file does not exist write header first
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Results saved to {filename}\n")

    #printing
    print(f"experiment_id: {experiment_id}")
    print(f"timestamp: {timestamp}")
    print(f"experiment_name: {experiment_name}")

    print(f"test_accuracy: {round(test_metrics['accuracy'], 4)}")
    print(f"test_precision: {round(test_metrics['precision'], 4)}")
    print(f"test_recall: {round(test_metrics['recall'], 4)}")
    print(f"test_f1_none_class0: {round(test_metrics['f1_none_class0'], 4)}")
    print(f"test_f1_none_class1: {round(test_metrics['f1_none_class1'], 4)}")
    print(f"test_f1_macro: {round(test_metrics['f1_macro'], 4)}")
    print(f"test_true_negative: {test_metrics['true_negative']}")
    print(f"test_false_positive: {test_metrics['false_positive']}")
    print(f"test_false_negative: {test_metrics['false_negative']}")
    print(f"test_true_positive: {test_metrics['true_positive']}")
    print(f"test_FAR: {test_metrics['FAR']}")
    print(f"test_FRR: {test_metrics['FRR']}")

    print(f"val_accuracy: {round(val_metrics['accuracy'], 4)}")
    print(f"val_precision: {round(val_metrics['precision'], 4)}")
    print(f"val_recall: {round(val_metrics['recall'], 4)}")
    print(f"val_f1_none_class0: {round(val_metrics['f1_none_class0'], 4)}")
    print(f"val_f1_none_class1: {round(val_metrics['f1_none_class1'], 4)}")
    print(f"val_f1_macro: {round(val_metrics['f1_macro'], 4)}")
    print(f"val_true_negative: {val_metrics['true_negative']}")
    print(f"val_false_positive: {val_metrics['false_positive']}")
    print(f"val_false_negative: {val_metrics['false_negative']}")
    print(f"val_true_positive: {val_metrics['true_positive']}")
    print(f"val_FAR: {val_metrics['FAR']}")
    print(f"val_FRR: {val_metrics['FRR']}")

    print(f"train_accuracy: {round(train_metrics['accuracy'], 4)}")
    print(f"train_precision: {round(train_metrics['precision'], 4)}")
    print(f"train_recall: {round(train_metrics['recall'], 4)}")
    print(f"train_f1_none_class0: {round(train_metrics['f1_none_class0'], 4)}")
    print(f"train_f1_none_class1: {round(train_metrics['f1_none_class1'], 4)}")
    print(f"train_f1_macro: {round(train_metrics['f1_macro'], 4)}")
    print(f"train_true_negative: {train_metrics['true_negative']}")
    print(f"train_false_positive: {train_metrics['false_positive']}")
    print(f"train_false_negative: {train_metrics['false_negative']}")
    print(f"train_true_positive: {train_metrics['true_positive']}")
    print(f"train_FAR: {train_metrics['FAR']}")
    print(f"train_FRR: {train_metrics['FRR']}")

    print(f"lr: {lr}")
    print(f"batch_size: {batch_size}")
    print(f"max_epochs: {max_epochs}")
    print(f"best_epoch: {best_epoch}")


def get_experiment_id(filename):
    if not os.path.isfile(filename):
        return 1
    with open(filename, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

if __name__ == "__main__":
    train()