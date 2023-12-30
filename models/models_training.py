import ../common.py


def fit_epoch(model, x_train, y_train, criterion, optimizer, gyro_only=False):
    '''
    The function that trains a deep-learning model for one epoch on train dataset
    model: deep-learning model
    x_train: training dataset
    y_train: training ground truth
    criterion: loss function
    optimizer: torch optimizer
    gyro_only: if False, then the model takes accelerometer and gyroscope measurements

    returns: train loss
    '''
    running_loss = 0.0
    processed_data = 0
    batch_n=0
    model.train()
    for b_x, b_y in zip(x_train, y_train):
        if not(gyro_only):
           b_acc = b_x[:, 3:6].to(DEVICE)
        b_gyro = b_x[:, 0:3].to(DEVICE)
        b_y = b_y.to(DEVICE)
        optimizer.zero_grad()
        if not(gyro_only):
           b_out = model(b_acc, b_gyro)
        else:
           b_out = model(b_gyro)
        loss = criterion(b_out, b_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * b_acc.size(0)
        processed_data += b_y.size(0)
        if not(gyro_only):
           del b_acc
        del b_gyro
        del b_y
        del loss
        torch.cuda.empty_cache()
    train_loss = running_loss / processed_data
    return train_loss

def eval_epoch(model, x_val, y_val, criterion, gyro_only=False):
    '''
    The function that trains a deep-learning model for one epoch on train dataset
    model: deep-learning model
    x_val: validation dataset
    y_val: validation ground truth
    criterion: loss function
    gyro_only: if False, then the model takes accelerometer and gyroscope measurements

    returns: validation loss
    '''
    model.eval()
    running_loss = 0.0
    processed_size = 0
    for b_x, b_y in zip(x_val, y_val):
        if not(gyro_only):
           b_acc = b_x[:, 3:6].to(DEVICE)
        b_gyro = b_x[:, 0:3].to(DEVICE)
        b_y = b_y.to(DEVICE)
        with torch.set_grad_enabled(False):
            if not(gyro_only):
               b_out = model(b_acc, b_gyro)
            else:
               b_out = model(b_gyro)
            loss = criterion(b_out, b_y)
        running_loss += loss.item() * b_acc.size(0)
        processed_size += b_y.size(0)
        if not(gyro_only):
           del b_acc
        del b_gyro
        del b_y
        del loss
        torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    return val_loss

def train(x_train, x_val, y_train, y_val, model, criterion, epochs, opt, sched=None, start_epoch=0, gyro_only=False):
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
    gyro_only: if False, then the model takes accelerometer and gyroscope measurements

    returns: train and validation losses values
    '''
    history = []
    log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f}, val_loss: {v_loss:0.4f}, LR = {lr:.2e}, elapsed time = {e_time} (+{ep_time:1d} sec)"

    start = time.process_time()
    for epoch in range(epochs):
        ep_start = time.process_time()
        train_loss = fit_epoch(model, x_train, y_train, criterion, opt, gyro_only)
        val_loss = eval_epoch(model, x_val, y_val, criterion, gyro_only)
        history.append((train_loss, val_loss))
        if sched is not None:
            sched.step(val_loss)
        end = time.process_time()
        etime = round(end - start)
        ep_time = round(end - ep_start)
        print(log_template.format(ep=epoch+1+start_epoch, t_loss=train_loss, v_loss=val_loss, lr=optimizer.param_groups[0]['lr'], \
                                           e_time=datetime.timedelta(seconds=etime), ep_time=ep_time))
    return np.array(history)