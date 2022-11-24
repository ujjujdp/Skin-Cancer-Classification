###Imports
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

#### Methods for Threat Score
def getCM(true_y, pred_y):
    true_y = true_y
    pred_y = pred_y
    return confusion_matrix(true_y, pred_y,labels=[0,1,2])

def getTS(tp,tn,fp,fn):
    return tp/(tp+fp+fn)

def calcTS(true_y, pred_y):
    
    confusion_metric = (getCM(true_y,pred_y)).T
    
    mel_TP = confusion_metric[0][0]
    mel_TN = confusion_metric[1][1] + confusion_metric[1][2] + confusion_metric[2][1] + confusion_metric[2][2]
    mel_FP = confusion_metric[0][1] + confusion_metric[0][2] 
    mel_FN = confusion_metric[1][0] + confusion_metric[2][0]
  
    nev_TP = confusion_metric[1][1]
    nev_TN = confusion_metric[0][0] + confusion_metric[2][0] + confusion_metric[0][2] + confusion_metric[2][2]
    nev_FP = confusion_metric[1][0] + confusion_metric[1][2] 
    nev_FN = confusion_metric[0][1] + confusion_metric[2][1]

    sk_TP = confusion_metric[2][2]
    sk_TN = confusion_metric[0][0] + confusion_metric[0][1] + confusion_metric[1][0] + confusion_metric[1][1]
    sk_FP = confusion_metric[2][0] + confusion_metric[2][1] 
    sk_FN = confusion_metric[0][2] + confusion_metric[1][2]
    
    mel_TS = getTS(mel_TP,mel_TN,mel_FP,mel_FN)
    nev_TS = getTS(nev_TP,nev_TN,nev_FP,nev_FN)
    sk_TS = getTS(sk_TP,sk_TN,sk_FP,sk_FN)
    
    return (mel_TS + nev_TS + sk_TS)/3

### Training method
def train(epochs, loaders, model, optimizer, criterion, device, save_path, verbose=True,use_saved_model=False):
    
    if(use_saved_model == True):
        saved_model = torch.load(save_path)
        saved_model.eval()
        print("Saved Model Returned")
        return saved_model
      
    ### Moving model to hardware accelerator
    model.to(device)

    valid_loss_min = 10 #Lowest Achieved.

    print("Training starts...\n")

    ### Training and Validation Loop
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        print("Epochs Number:", e+1)

        # Train loop
        n_bactches = len(loaders['train'])
        model.train()
        for batch_idx, (feature, label) in enumerate(loaders['train']):
            feature, label = feature.to(device), label.to(device)
            print("\tTrain Batch Number: [", batch_idx,", ",n_bactches,"]")
            optimizer.zero_grad()

            log_ps, aux_outputs = model(feature)
            loss1 = criterion(log_ps, label)
            loss2 = criterion(aux_outputs, label)
            loss = loss1 + 0.4 * loss2

            loss.backward()
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.item() - train_loss)
        
        # Validation loop
        n_bactches = len(loaders['valid'])
        model.eval()
        for batch_idx, (feature, label) in enumerate(loaders['valid']):
            feature, label = feature.to(device), label.to(device)
            print("\tValidation Batch Number: [", batch_idx,", ",n_bactches,"]")

            log_ps = model(feature)
            loss = criterion(log_ps, label)

            valid_loss += (1 / (batch_idx + 1)) * (loss.item() - valid_loss)

        # Print out results
        if verbose:
            print('Epoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(e + 1, train_loss, valid_loss))
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            print('Saving Model...')
    
    print("Training ends...\n")
    return model

### Testing method
def test(loaders, model, criterion, device, verbose=True):

    ### Moving model to hardware accelerator
    model.to(device)

    # Initializing test results
    test_loss = 0.0
    correct = 0
    total = 0
    
    true_labels = []
    pred_labels = []
    
    print("Testing starts...\n")
    # Test loop
    model.eval()
    for batch_idx, (feature, label) in enumerate(loaders['test']):
        feature, label = feature.to(device), label.to(device)

        log_ps = model(feature)
        loss = criterion(log_ps, label)

        test_loss += (1 / (batch_idx + 1)) * (loss.item() - test_loss)

        pred = log_ps.data.max(1, keepdim=True)[1]
        correct += np.sum((label.t()[0] == pred).cpu().numpy())
        total += label.shape[0]
        
        true_labels.extend(label.tolist())
        pred_labels.extend(pred)

    pred_la = [i.detach().cpu().numpy().tolist()[0] for i in pred_labels]
    
    threat_score = calcTS(true_labels, pred_la)
    
    if verbose:
        print('Test Loss: {}\n'.format(test_loss))
        print('Test Accuracy: {}%, ({}/{})'.format(100 * correct / total, correct, total))
        print('Threat Score: {}%'.format(threat_score))

    print("Testing ends...\n")

"""
Threat Score =  TP/(TP+FN+FP)
"""