import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from src.model import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
import csv
import os

from src.resnet_model import ResNet18
from src.train_util import time_mask, time_shift, freq_mask, get_threshold_roc, gauss_noise
from datetime import datetime
from preprocessing_pipeline.util import random_data_split


def augment_batch(x, labels, sr=22050):
    """
    Apply random data augmentation techniques to a batch of spectrograms.

    Arguments:
    x (torch.Tensor): Input spectrogram batch of shape (batch_size, channels, time, freq).
    labels (torch.Tensor): Labels for the batch.
    sr (int, optional): Sample rate in Hz. Default is 22050. Not used in augmentations.
    
    Returns:
    torch.Tensor: Augmented spectrogram batch with same shape as input.
    """
    # Randomly apply time masking 
    if torch.rand((), device=x.device).item() < 0.5:
        x = time_mask(x, max_width=10)
    
    # Randomly apply frequency masking 
    if torch.rand((), device=x.device).item() < 0.5:
        x = freq_mask(x, max_height=5)
    
    # Randomly apply time shifting 
    if torch.rand((), device=x.device).item() < 0.5:
        x = time_shift(x, max_shift=5)
   
    return x

def train():
    """
    This is our final training loop.
    
    This function consists of the complete training pipeline:
    1. Load preprocessed training, validation, and test data
    2. Initialize model, optimizer, and loss criterion
    3. Train for specified number of epochs with validation after each epoch
    4. Save best model based on validation loss
    5. Evaluate final model on test set
    6. Log all results and metrics to CSV and TensorBoard
    
    IMPORTANT NOTE: Must be run from IML_voice_recognition directory using 'python -m src.train' to ensure correct relative imports and results file paths.
    """
    
    # Model, inputs and labels must all be on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessed data (already split into train/val/test sets)
    data = random_data_split(path="./preprocessing_pipeline")
    x_train, y_train = data['train']
    x_valid, y_valid = data['validate']
    x_test, y_test = data['test']

    # Extract speaker labels for the speaker confusion matrix later
    test_labels = data['test_labels']
    print(f"Test speaker labels length: {len(test_labels)}")
    validate_labels = data['validate_labels']
    print(f"Validate speaker labels length: {len(validate_labels)}")
    print(f"Test labels all length: {len(y_test)}")
    print(f"Validate labels all length: {len(y_valid)}")

    x_train = x_train.float()
    x_valid = x_valid.float()
    x_test = x_test.float()
    batch_size =32
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)
    
    # Use weighted cross-entropy loss to handle class imbalance
    weights = torch.tensor([1.0, data['weight']]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    

    model_name = "resnet_trial_221_32_1_dropout_0.5_adam_lr_0.0005_wd_0.01_conv_avgpool"
    net = ResNet18().to(device)

    # Experiment notes for tracking model changes
    notes = "avg pool in first conv layer instead of max pool"

   
    learning_rate = 0.0005
    weight_dec = 0.01
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_dec)

    best_model_loss = float('inf')

    #directory for results csv file
    results_directory = "./results"
    os.makedirs(results_directory, exist_ok=True)
    filename_results = "./results/tests_results.csv"
    filename_speakers_conf = "./results/speakers_conf_matrix.csv"

    # Experiment ID is auto-incremented based on number of existing experiments
    experiment_id = get_experiment_id(filename_results)

    # directory with models
    os.makedirs("./models", exist_ok=True)
    
    # setup TensorBoard logging
    writer = SummaryWriter(log_dir=f"./{get_tensorboard_logdir()}/id_{experiment_id}_{model_name}")

    max_epochs = 50
    best_epoch = 0
    print("TRAINING START")
    for epoch in range(max_epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.to(device)
            inputs = augment_batch(inputs, labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            current_batch_num = epoch * len(train_loader) + i

        # save the best model yet
        val_loss = validate(net, criterion, val_loader, device)
        
        # log training and validation loss to TensorBoard
        writer.add_scalars('loss', {
                'train': running_loss/len(train_loader),
                'validation': val_loss
            }, epoch)
        
        # save model if validation loss improves
        if val_loss < best_model_loss:
            best_model_loss = val_loss
            best_epoch = epoch
            model_path = f"./models/id_{experiment_id}_{model_name}.pth"
            torch.save({'epoch': epoch, 'net_state_dict': net.state_dict(),'val_loss': val_loss }, model_path)
            print(f"BEST (val_loss: {val_loss:.4f}) epoch:{epoch}")
        else:
            print(f"No improvement: val_loss: {val_loss:.4f} epoch:{epoch}  best:{best_epoch}")

            # return to training mode after validation
            net.train()

        print(f"Epoch {epoch}, loss: {running_loss/len(train_loader):.3f}")
       

    threshold, auc = get_threshold_roc(net, val_loader, device)
    print(f"Testing the best model, AUC: {auc:.4f}")
    
    # FINAL EVALUATION
    net.load_state_dict((torch.load(f"./models/id_{experiment_id}_{model_name}.pth", map_location=device))['net_state_dict'])
    test_metrics = calculate_metrics(net, test_loader, device, test_labels, threshold)
    val_metrics = calculate_metrics(net, val_loader, device, validate_labels, threshold)
    train_metrics = calculate_metrics(net, train_loader, device, threshold=threshold)
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
                                   notes=notes
                                   )

    # Save per-speaker confusion matrix
    save_to_csv_speaker_confusion_matrix(filename_speakers_conf,
                                         experiment_id,
                                         model_name,
                                         test_metrics['results_speakers'],
                                         val_metrics['results_speakers'])

    print("TEST END")
    writer.close()


def validate(net, criterion, valloader: DataLoader, device):
    """
    This function runs the model in evaluation mode
    and computes validation loss along with per-class accuracy metrics.
    
    Returns:
    float: Average validation loss across all batches.
    """
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
        print(f"\nVAL Loss: {avg_loss:.4f} | Class 0: {acc0:.1f} | Class 1: {acc1:.1f}")
        return avg_loss


def calculate_metrics(net: SimpleCNN, dataloader: DataLoader, device, speakers_labels=[], threshold=0.5):
    """
    Calculate comprehensive metrics to track results across train, test and validation data and across speakers -
    (creates a confusion matrix for class 1 speakers).
    
    
    Args:
    - net: The neural network model to evaluate.
    - dataloader: DataLoader providing input batches.
    - device: Device to run computations on (CPU or GPU).
    - speakers_labels: List of speaker identifiers for per-speaker metrics.
    - threshold: classification threshold. 
    
    Returns dictionary containing:
    - accuracy, precision, recall
    - f1_none_class0, f1_none_class1: per-class F1 scores
    - f1_macro: unweighted mean F1 across classes
    - true_negative, false_positive, false_negative, true_positive: confusion matrix elements
    - FAR (False Acceptance Ratio)
    - FRR (False Rejection Ratio)
    - results_speakers (dict): per-speaker TP and FN ratios (for class 1 speakers)
    """
    all_predictions = []
    all_labels = []
    net.eval()
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
           # _,predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probs > threshold).long()
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

    results_speakers = {}


    speaker_to_id = {name: i for i, name in enumerate(sorted(set(speakers_labels)))}
    speakers_labels_ids = [speaker_to_id[name] for name in speakers_labels]
    speakers_labels_ids = torch.tensor(speakers_labels_ids)

 
    if not len(speakers_labels)==0:
        speakers_class1 = ['kasia', 'kuba', 'marcin', 'sylwia']
        all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.int64)
        for sp in speakers_class1:
            sp_id = speaker_to_id[sp]
            # get samples belonging to this speaker
            sth = (speakers_labels_ids == sp_id)
            sp_pred = all_predictions_tensor[sth]
            print(f"Speaker: {sp} predictions: ")
            print(sp_pred)
            print('\n')

            TP = (sp_pred == 1).sum().item()
            print(f"TP for {sp}: {TP}\n")
            FN = (sp_pred == 0).sum().item()
            print(f"FN for {sp}: {FN}\n")

            results_speakers[sp] = {
                "TP": round(TP/(TP+FN),4),
                "FN": round(FN/(TP+FN),4),
            }



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
               'FRR': FRR,
               'results_speakers':results_speakers
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
    '''
    Function that saves the experiment results to a CSV file.
    '''

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


def save_to_csv_speaker_confusion_matrix(
    filename,
    experiment_id,
    experiment_name,
    test_results_speakers,
    val_results_speakers,
    notes=None,
):
    '''
    Function that saves the per-speaker confusion matrix results to a CSV file.
    '''

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    header = [
        "experiment_id",
        "timestamp",
        "experiment_name",
        "test_kasia_TP",
        "test_kasia_FN", 
        "test_kuba_TP",
        "test_kuba_FN",
        "test_marcin_TP",
        "test_marcin_FN",
        "test_sylwia_TP",
        "test_sylwia_FN",
        "val_kasia_TP",
        "val_kasia_FN", 
        "val_kuba_TP",
        "val_kuba_FN",
        "val_marcin_TP",
        "val_marcin_FN",
        "val_sylwia_TP",
        "val_sylwia_FN",
        "notes"
    ]

    row = [
        experiment_id,
        timestamp,
        experiment_name,
        test_results_speakers['kasia']['TP'],
        test_results_speakers['kasia']['FN'],
        test_results_speakers['kuba']['TP'],
        test_results_speakers['kuba']['FN'],
        test_results_speakers['marcin']['TP'],
        test_results_speakers['marcin']['FN'],
        test_results_speakers['sylwia']['TP'],
        test_results_speakers['sylwia']['FN'],
        val_results_speakers['kasia']['TP'],
        val_results_speakers['kasia']['FN'],
        val_results_speakers['kuba']['TP'],
        val_results_speakers['kuba']['FN'],
        val_results_speakers['marcin']['TP'],
        val_results_speakers['marcin']['FN'],
        val_results_speakers['sylwia']['TP'],
        val_results_speakers['sylwia']['FN'],
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


def get_experiment_id(filename):
    '''
    File to track how many experiments have been run so far to get unique experiment ID.
    '''
    if not os.path.isfile(filename):
        return 1
    with open(filename, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def get_tensorboard_logdir(base="tensor_board_outputs", max_items=100):
    import re
    if os.path.exists(base):
        parent = "."
        rotated = [
            d for d in os.listdir(parent)
            if re.fullmatch(rf"{re.escape(base)}(_(\d+))?$", d)
               and os.path.isdir(os.path.join(parent, d))
        ]
        rotated.sort()
        latest = rotated[-1]
        latest_path = os.path.join(parent, latest)

        subdirs = [d for d in os.listdir(latest_path)
                   if os.path.isdir(os.path.join(latest_path, d))]

        if len(subdirs) < max_items:
            return latest_path

        new_dir = f"{base}_{len(rotated)}"
        os.makedirs(new_dir, exist_ok=True)
        return new_dir
    else:
        os.makedirs(base, exist_ok=True)
    return base

if __name__ == "__main__":
    train()