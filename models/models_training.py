import numpy as np
import torch
import time
import datetime

# SEED = 0

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)


def fit_epoch_ag(model, x_train, y_train, criterion, optimizer, DEVICE):
   '''
   The function that trains a deep-learning model for one epoch on train dataset
   model: deep-learning model
   x_train: training dataset
   y_train: training ground truth
   criterion: loss function
   optimizer: torch optimizer

   returns: train loss
   '''
   running_loss = 0.0
   processed_data = 0
   model.train()
   for b_x, b_y in zip(x_train, y_train):
      b_acc = b_x[:, 3:6].to(DEVICE)
      b_gyro = b_x[:, 0:3].to(DEVICE)
      b_y = b_y.to(DEVICE)
      optimizer.zero_grad()
      b_out = model(b_acc, b_gyro)
      loss = criterion(b_out, b_y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * b_gyro.size(0)
      processed_data += b_y.size(0)
      del b_acc
      del b_gyro
      del b_y
      del loss
      torch.cuda.empty_cache()
   train_loss = running_loss / processed_data
   return train_loss


def eval_epoch_ag(model, x_val, y_val, criterion, DEVICE):
    '''
    The function that trains a deep-learning model for one epoch on train dataset
    model: deep-learning model
    x_val: validation dataset
    y_val: validation ground truth
    criterion: loss function

    returns: validation loss
    '''
    model.eval()
    running_loss = 0.0
    processed_size = 0
    for b_x, b_y in zip(x_val, y_val):
         b_acc = b_x[:, 3:6].to(DEVICE)
         b_gyro = b_x[:, 0:3].to(DEVICE)
         b_y = b_y.to(DEVICE)
         with torch.set_grad_enabled(False):
            b_out = model(b_acc, b_gyro)
            loss = criterion(b_out, b_y)
         running_loss += loss.item() * b_gyro.size(0)
         processed_size += b_y.size(0)
         del b_acc
         del b_gyro
         del b_y
         del loss
         torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    return val_loss

def fit_epoch_gm(model, x_train, y_train, criterion, optimizer, DEVICE):
   '''
   The function that trains a deep-learning model for one epoch on train dataset
   model: deep-learning model
   x_train: training dataset
   y_train: training ground truth
   criterion: loss function
   optimizer: torch optimizer

   returns: train loss
   '''
   running_loss = 0.0
   processed_data = 0
   model.train()
   for b_x, b_y in zip(x_train, y_train):
      b_mag = b_x[:, 6:9].to(DEVICE)
      b_gyro = b_x[:, 0:3].to(DEVICE)
      b_y = b_y.to(DEVICE)
      optimizer.zero_grad()
      b_out = model(b_mag, b_gyro)
      loss = criterion(b_out, b_y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * b_gyro.size(0)
      processed_data += b_y.size(0)
      del b_mag
      del b_gyro
      del b_y
      del loss
      torch.cuda.empty_cache()
   train_loss = running_loss / processed_data
   return train_loss

def eval_epoch_gm(model, x_val, y_val, criterion, DEVICE):
    '''
    The function that trains a deep-learning model for one epoch on train dataset
    model: deep-learning model
    x_val: validation dataset
    y_val: validation ground truth
    criterion: loss function

    returns: validation loss
    '''
    model.eval()
    running_loss = 0.0
    processed_size = 0
    for b_x, b_y in zip(x_val, y_val):
         b_mag = b_x[:, 6:9].to(DEVICE)
         b_gyro = b_x[:, 0:3].to(DEVICE)
         b_y = b_y.to(DEVICE)
         with torch.set_grad_enabled(False):
            b_out = model(b_mag, b_gyro)
            loss = criterion(b_out, b_y)
         running_loss += loss.item() * b_gyro.size(0)
         processed_size += b_y.size(0)
         del b_mag
         del b_gyro
         del b_y
         del loss
         torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    return val_loss

def fit_epoch_g(model, x_train, y_train, criterion, optimizer, DEVICE):
   '''
   The function that trains a deep-learning model for one epoch on train dataset
   model: deep-learning model
   x_train: training dataset
   y_train: training ground truth
   criterion: loss function
   optimizer: torch optimizer

   returns: train loss
   '''
   running_loss = 0.0
   processed_data = 0
   model.train()
   for b_x, b_y in zip(x_train, y_train):
      b_gyro = b_x[:, 0:3].to(DEVICE)
      b_y = b_y.to(DEVICE)
      optimizer.zero_grad()
      b_out = model(b_gyro)
      loss = criterion(b_out, b_y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * b_gyro.size(0)
      processed_data += b_y.size(0)
      del b_gyro
      del b_y
      del loss
      torch.cuda.empty_cache()
   train_loss = running_loss / processed_data
   return train_loss

def eval_epoch_g(model, x_val, y_val, criterion, DEVICE):
    '''
    The function that trains a deep-learning model for one epoch on train dataset
    model: deep-learning model
    x_val: validation dataset
    y_val: validation ground truth
    criterion: loss function

    returns: validation loss
    '''
    model.eval()
    running_loss = 0.0
    processed_size = 0
    for b_x, b_y in zip(x_val, y_val):
         b_gyro = b_x[:, 0:3].to(DEVICE)
         b_y = b_y.to(DEVICE)
         with torch.set_grad_enabled(False):
            b_out = model(b_gyro)
            loss = criterion(b_out, b_y)
         running_loss += loss.item() * b_gyro.size(0)
         processed_size += b_y.size(0)
         del b_gyro
         del b_y
         del loss
         torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    return val_loss

def train(x_train, x_val, y_train, y_val, model, criterion, epochs, opt, DEVICE, sched=None, start_epoch=0):
   '''
   training function for a model
   x_train: training dataset
   x_val: validation dataset
   y_train: training ground truth
   y_val: validation ground truth
   model: deep-learning model
   criterion: torch loss function
   epochs: number of epochs
   opt: torch optimizer
   sched: torch scheduler
   start_epoch: starting epoch number, only influences output

   returns: train and validation losses values
   '''
   history = []
   log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f}, val_loss: {v_loss:0.4f}, LR = {lr:.2e}, elapsed time = {e_time} (+{ep_time:1d} sec)"

   start = time.process_time()
   for epoch in range(epochs):
      ep_start = time.process_time()
      if model.magnetic:
         # Training GM-DoorINet
         train_loss = fit_epoch_gm(model, x_train, y_train, criterion, opt, DEVICE)
         val_loss = eval_epoch_gm(model, x_val, y_val, criterion, DEVICE)   
      elif not(model.gyro_only):
         # Training AG-DoorINet
         train_loss = fit_epoch_ag(model, x_train, y_train, criterion, opt, DEVICE)
         val_loss = eval_epoch_ag(model, x_val, y_val, criterion, DEVICE)
      else:
         # Training G-DoorINet
         train_loss = fit_epoch_g(model, x_train, y_train, criterion, opt, DEVICE)
         val_loss = eval_epoch_g(model, x_val, y_val, criterion, DEVICE)
      history.append((train_loss, val_loss))
      if sched is not None:
         sched.step(val_loss)
      end = time.process_time()
      etime = round(end - start)
      ep_time = round(end - ep_start)
      print(log_template.format(ep=epoch+1+start_epoch, t_loss=train_loss, v_loss=val_loss, lr=opt.param_groups[0]['lr'], \
                                           e_time=datetime.timedelta(seconds=etime), ep_time=ep_time))
   return np.array(history)