import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm_notebook
from collections import defaultdict
import copy
import warnings
warnings.simplefilter("ignore")

from utils.scorer import Scorer


def write_text(writer, model_name, history, epoch=0, batch=None):
    text_with_metrics = ''
    for metric in history:
        text_with_metrics += metric + ': ___'+ str(history[metric]) + '___;   '
    if batch is None:
        writer.add_text(model_name + '/epoch_' + str(epoch), text_with_metrics, batch)
    else:
        writer.add_text(model_name, text_with_metrics, epoch)


def write_scalars(writer, model_name, history, n_epoch, epoch):
    writer.add_scalars(model_name + ' loss', {'train': history['loss_train'],\
                                                               'val': history['loss_val']}, epoch)
    for metric in history:
        writer.add_scalar(f'{model_name}/{metric}', history[metric], epoch)


def print_results(model_name, history, epoch=None):
    print(model_name)
    if epoch is not None:
        print(f'epoch: {epoch}   train loss: {history["loss_train"]}   val loss: {history["loss_val"]}')
    for metric in history:
        if 'loss' not in metric:
            print(f'{metric}: {history[metric]}')


class Trainer():
    """
    метод train - обучение сетей и сбор статистики по метрикам на этапе обучения
    метод test - inference модели и подсчет метрик, если передаются размеченные данные
    """
    def __init__(self, model, device='cpu', label_names=None, model_name='model'):
        self.model = model
        self.model_name = model_name
        self.label_names = label_names
        self.device = device
        self.loss = None

    def train(self, train_dataloader=None, val_dataloader=None, n_epoch=10, optim=Adam, weight_decay=0.0,\
              schedul=None, loss=torch.nn.CrossEntropyLoss, weighted=True, lr=2e-5, show_results=True, \
              saved_models_dir=None, verbose=True, early_stopping=True, max_gap=2, gamma=None):
        params = locals()

        scorer = Scorer()
        writer = SummaryWriter(comment='_'+self.model_name)
        total_history = []

        param_text = ''
        for param in params:
            if param != 'train_data' and param != 'train_labels' and param !='self':
                param_text += param + ': ___'+ str(params[param]) + '___;   '
        writer.add_text(self.model_name + ' train parameters ', param_text)
            
        self.model.to(self.device)
        
        self.loss = loss            
              
        optimizer = optim(self.model.parameters(), weight_decay=weight_decay, lr=lr)
        if not schedul is None:
            scheduler = schedul(optimizer)
        
        prev_model = None
        best_loss = 0.
                 
        for epoch in range(n_epoch):    
            loss_train_lst = []
            self.model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm_notebook(train_dataloader, disable=not verbose)):
                optimizer.zero_grad()
                  
                batch_pred = self.model(x_batch.to(self.device))
                
                batch_loss = self.loss(batch_pred, y_batch.to(self.device))

                loss_train_lst.append(batch_loss.item())
                writer.add_scalar(self.model_name + ' non-binary/batch loss/train ' + 'epoch ' + str(epoch), loss_train_lst[-1], batch_idx)

                batch_loss.backward()
                optimizer.step()
                if not schedul is None:
                    scheduler.step()

            lbl_val_lst = []
            pred_val_lst = []
            pred_proba_val_lst = []
            loss_val_lst = []
            
            dice_lst = []
    
            self.model.eval()
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm_notebook(val_dataloader, disable=not verbose)):
                with torch.no_grad():
                    batch_pred = self.model(x_batch.to(self.device))
                    batch_loss = self.loss(batch_pred, y_batch.to(self.device))
                    loss_val_lst.append(batch_loss.item())
                    for mask_true, mask_pred in zip(y_batch.cpu(), batch_pred.cpu()):
                        dice_lst.append(scorer(mask_true, mask_pred))    

            history = {}
            
            history.update(pd.DataFrame(dice_lst).mean().apply(lambda x: np.round(x, 4)).to_dict())
            history['loss_train'] = np.round(np.mean(loss_train_lst), 4)
            history['loss_val'] = np.round(np.mean(loss_val_lst), 4)
            total_history.append(history)
            write_scalars(writer, self.model_name, history, n_epoch=n_epoch, epoch=epoch)
            write_text(writer, self.model_name, history, epoch=epoch)

            if show_results:
                print_results(self.model_name, history, epoch)

            if not saved_models_dir is None:    
                torch.save(self.model.state_dict(), saved_models_dir + self.model_name + '_' + str(epoch) + '.pth')         
  
            if epoch == 0:
                best_epoch = 0
                patience = 0
                best_loss = total_history[0]['loss_val']
                if not saved_models_dir is None:
                    prev_model = saved_models_dir + f'{self.model_name}_tmp.pth'
                    torch.save(self.model.state_dict(), prev_model)
                else:
                    if self.device!='cpu':
                        self.model.to('cpu')
                    prev_model = copy.deepcopy(self.model)
                    if self.device!='cpu':
                        self.model.to(self.device)         

            if epoch >= 1:
                if total_history[epoch]['loss_val'] < best_loss:
                    best_epoch = epoch
                    patience = 0
                    best_loss = total_history[epoch]['loss_val']
                    if not saved_models_dir is None:
                        prev_model = saved_models_dir + f'{self.model_name}_tmp.pth'
                        torch.save(self.model.state_dict(), prev_model)
                    else:
                        if self.device!='cpu':
                            self.model.to('cpu')
                        prev_model = copy.deepcopy(self.model)
                        if self.device!='cpu':
                            self.model.to(self.device)
                else:
                    patience += 1
                    if patience == max_gap:
                        if not saved_models_dir is None:
                            self.model.load_state_dict(torch.load(prev_model, map_location=self.device))
                        else:    
                            self.model = prev_model
                            if self.device!='cpu':
                                self.model.to(self.device)
                        if not gamma is None:
                            for g in optimizer.param_groups:
                                g['lr'] = g['lr']*gamma   
                            if not saved_models_dir is None:
                                self.model.load_state_dict(torch.load(prev_model, map_location=self.device))
                            else:    
                                self.model = prev_model
                                if self.device!='cpu':
                                    self.model.to(self.device)
                            patience = 0
                        elif early_stopping==True:
                            if not saved_models_dir is None:
                                self.model.load_state_dict(torch.load(prev_model, map_location=self.device))
                            else:    
                                self.model = prev_model
                                if self.device!='cpu':
                                    self.model.to(self.device)
                            break_flag = True
                            break

        if not saved_models_dir is None:
            torch.save(self.model.state_dict(), saved_models_dir + self.model_name + '_best_' + str(best_epoch) + '.pth')
        writer.close()
        return total_history[best_epoch]

    
    def test(self, test_dataloader, show_results=True, verbose=True, log=True):
        if log:
            writer = SummaryWriter(comment='_' + self.model_name + '_test')
        scorer = Scorer()
              
        loss_test_lst=[]
        lbl_test_lst = []
        pred_test_lst = []
        pred_proba_test_lst =[]

        self.model.to(self.device)
        self.model.eval()
        
        y_batch=None
              
        for batch_idx, batch in enumerate(tqdm_notebook(test_dataloader, disable=not verbose)):    
            with torch.no_grad():
                if not isinstance(batch, list):
                    x_batch = batch
                else:
                    x_batch, y_batch = batch

                batch_pred = self.model(x_batch.to(self.device))

                if not y_batch is None:
                    lbl_test_lst.append(y_batch.numpy().reshape(-1,1))
            
                pred_test_lst.append(torch.argmax(batch_pred, dim=1).cpu().reshape(-1, 1))
                pred_proba_test_lst.append(batch_pred.cpu().numpy())

        if not y_batch is None:
            lbl_test = test_dataloader.dataset.labels
        
        pred_test = np.vstack(pred_test_lst)
        pred_proba_test = np.vstack(pred_proba_test_lst)
        pred_proba_test = np.exp(pred_proba_test)/np.sum(np.exp(pred_proba_test), axis=1).reshape(-1, 1)
 
        if not y_batch is None:
            history = scorer(pred_test, pred_proba_test, lbl_test)
            if show_results:
                print_results(model_name=self.model_name, history=history)
            if log:
                write_text(writer, self.model_name + '_test', history)
                writer.close()
            return pred_test, pred_proba_test, history
        return pred_test, pred_proba_test
