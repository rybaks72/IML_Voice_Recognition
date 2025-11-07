import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from model import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from datetime import datetime


# fake test dataset for testing,  I assume size 64x64 for spectrogram, but we can change it quickly later
class FakeSpectrogramDataset(Dataset):
    def __init__(self, n_samples=200):
        self.data = np.random.randn(n_samples, 1, 64, 64).astype(np.float32)
        self.labels = np.random.randint(0, 2, n_samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def train():
    #model, inputs and labels have to be on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = FakeSpectrogramDataset()
    batch_size = 8
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_val = FakeSpectrogramDataset(n_samples=30)
    valloader = DataLoader(dataset_val, batch_size=8, shuffle=False)
    dataset_test = FakeSpectrogramDataset(n_samples=60)
    testloader = DataLoader(dataset_test, batch_size=8, shuffle=False)

    model_name = "first_trial"
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    results_directory = "..\\results"
    os.makedirs(results_directory, exist_ok=True)
    filename_results = "..\\results\\tests_results.csv"
    experiment_id = get_experiment_id(filename_results)
    best_model = float('inf')

    os.makedirs("../models", exist_ok=True)

    writer = SummaryWriter(log_dir=f"../tensor_board_outputs/fake_tests/id_{experiment_id}")

    print("TRAINING START")
    for epoch in range(3):  # loop over the dataset multiple times, this should be adjusted later
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            current_batch_num = epoch * len(trainloader) + i

            val_loss = validate(net, criterion, valloader, device)

            # one plot with both
            writer.add_scalars('loss', {
                'train': loss.item(),
                'validation': val_loss
            }, current_batch_num)

            #save the best model yet TODO does this criterion make sense
            if val_loss < best_model:
                best_model = val_loss

                model_path = f"../models/{model_name}_id_{experiment_id}.pth"
                torch.save({'epoch': epoch, 'batch': i, 'batch_num': current_batch_num,
                            'net_state_dict': net.state_dict(),'val_loss': val_loss }, model_path)
                print(f"BEST (val_loss: {val_loss:.4f}) epoch:{epoch} batch:{i} batch_num:{current_batch_num} train_loss:{loss.item():.4f}")
            else:
                print(f"No improvement: val_loss: {val_loss:.4f} epoch:{epoch} batch:{i} batch_num:{current_batch_num} train_loss:{loss.item():.4f}")

            # return to training mode after validation
            net.train()

        print(f"Epoch {epoch+1}, loss: {running_loss/len(trainloader):.3f}")

    print("Testing the best model")
    net.load_state_dict((torch.load(f"../models/{model_name}_id_{experiment_id}.pth", map_location=device))['net_state_dict'])
    test_results = test(net, testloader, device)
    log_experiment_results(filename_results,
                           experiment_id,
                           model_name,
                           test_results['accuracy'],
                           test_results['precision'],
                           test_results['recall'],
                           test_results['f1_none_class0'],
                           test_results['f1_none_class1'],
                           test_results['f1_micro'],
                           test_results['f1_macro'],
                           test_results['f1_weighted'],
                           test_results['f1_binary'],
                           test_results['true_negative'],
                           test_results['false_positive'],
                           test_results['false_negative'],
                           test_results['true_positive'],
                           lr = learning_rate,
                           batch_size= batch_size
                           )
    print("TEST END")

    writer.close()

    return test_results



def validate(net: SimpleCNN, criterion, valloader: DataLoader, device):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs,labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        avg_loss = total_loss / len(valloader)
        return avg_loss


def test(net: SimpleCNN, testloader: DataLoader, device):
    all_predictions = []
    all_labels = []
    net.eval()
    with torch.no_grad():
        for inputs,labels in testloader:
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

    #f1 = 2*TP/(2*TP+FP+FN) and with different parameter average:
    #metrics for each class are returned
    f1_none = f1_score(all_labels, all_predictions, average=None, labels=[0, 1])
    print(f1_none)

    f1_none_class0, f1_none_class1 = f1_none[0], f1_none[1]

    #Calculate metrics globally by counting the total true positives,
    # false negatives and false positives
    f1_micro = f1_score(all_labels, all_predictions, average='micro')

    #Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    f1_macro = f1_score(all_labels, all_predictions, average='macro')

    # Calculate metrics for each label, and find their average weighted by support
    # (the number of true instances for each label). This alters ‘macro’ to account
    # for label imbalance; it can result in an F-score that is not between precision and recall.
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

    #Only report results for the class specified by pos_label.
    # This is applicable only if targets (y_{true,pred}) are binary
    f1_binary = f1_score(all_labels, all_predictions, average='binary')

    results = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1_none_class0":f1_none_class0,
               "f1_none_class1": f1_none_class1,
               "f1_micro": f1_micro,
               "f1_macro": f1_macro,
               "f1_weighted": f1_weighted,
               "f1_binary": f1_binary,
               "true_negative": true_negative,
               "false_positive": false_positive,
               "false_negative": false_negative,
               "true_positive": true_positive
               }

    return results

def log_experiment_results(
    filename,
    experiment_id,
    experiment_name,
    accuracy,
    precision,
    recall,
    f1_none_class0,
    f1_none_class1,
    f1_micro,
    f1_macro,
    f1_weighted,
    f1_binary,
    true_negative,
    false_positive,
    false_negative,
    true_positive,
    lr,
    batch_size,
    notes=None,
):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    header = [
        "experiment_id",
        "timestamp",
        "experiment_name",
        "accuracy",
        "precision",
        "recall",
        "f1_none_class0",
        "f1_none_class1",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_binary",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "learning_rate",
        "batch_size",
        "notes"
    ]

    row = [
        experiment_id,
        timestamp,
        experiment_name,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1_none_class0, 4),
        round(f1_none_class1, 4),
        round(f1_micro, 4),
        round(f1_macro, 4),
        round(f1_weighted, 4),
        round(f1_binary, 4),
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        lr,
        batch_size,
        notes if notes is not None else ""
    ]

    print("\n" + "*" * 70)
    print(f" EXPERIMENT {experiment_id}: {experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 None class0: {f1_none_class0:.4f}")
    print(f"F1 none classs1: {f1_none_class1:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"F1 Binary: {f1_binary:.4f}")
    print(f"True Negative: {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"True Positive: {true_positive}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    if notes:
        print(f"Notes: {notes}")
    print("*" * 70 + "\n")


    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:

        writer = csv.writer(f)
        #if file does not exist write header first
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Results saved to {filename}\n")

def get_experiment_id(filename):
    if not os.path.isfile(filename):
        return 1
    with open(filename, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

if __name__ == "__main__":
    train()

