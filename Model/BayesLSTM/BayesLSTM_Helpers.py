from Helpers import *
from LSTM_Helpers import *
import os

path_LSTM ="/content/drive/MyDrive/Capstone/Model/BayesLSTM/"

def fetch_all_pred_real2(args, model, wg, rbscaler):    
    losses = []    
    with torch.no_grad():
      X =  next(iter(wg.all_data))[0].numpy()
      y =  next(iter(wg.all_data))[1].numpy()
      if args.gpu: # if gpu
          X = torch.from_numpy(X).cuda() # convert data to gpu
          y = torch.from_numpy(y).cuda() # convert target to gpu
      y_pred = model(X.float())
      med, qrg = rbscaler.get_fitted()
      target_median, qrange = med.values[-1], qrg[-1]
      y_pred = y_pred.cpu().detach().numpy()*qrange+target_median
      y_real = np.array([y*qrange+target_median for y in y.cpu().detach().numpy()]).reshape(y_pred.shape)
      return y_pred, y_real

def forecast(rbscaler, scaled_data, model, args, name = 'Manufacturing_value added_%_of_GDP',IW=4):
  med, qrg = rbscaler.get_fitted()
  target_median, qrange = med.values[-1], qrg[-1]
  feed = torch.from_numpy(np.expand_dims(scaled_data.iloc[-IW:, :], axis = 0))
  if args.gpu:
    feed = feed.cuda()
  y = model(feed.float())
  df =  pd.DataFrame({name: (y.cpu().detach().numpy()*qrange+target_median).tolist()[0]}, index =  [scaled_data.index[-1]+i for i in range(1, 6)])
  return df

def train(wg, args, model, rbscaler):

    # Save directory
    # Create the outputs folder if not created already
    if not os.path.exists(path_LSTM + args.experiment_name ):
        os.makedirs(path_LSTM + args.experiment_name )
    save_dir = path_LSTM + args.experiment_name  + "/output/"
    save_dir2 = path_LSTM + args.experiment_name  + "/result/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/Manufacturing')
        os.makedirs(save_dir + '/Service')
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
        os.makedirs(save_dir2 + '/Manufacturing')
        os.makedirs(save_dir2 + '/Service')

    learned_parameters = []
    # We only learn the last two layer with name containing classifier and freeze all the other weights
    ################ Code goes here ######################
    for name, param in model.named_parameters():
        # if name.startswith('classifier_layer'):
            learned_parameters.append(param)
    ######################################################

    # Adam only updates learned_parameters
    optimizer = torch.optim.Adam(learned_parameters, lr=args.learn_rate)

    if args.gpu:
        model.cuda()

    print("===== Training =====")
    start = time.time()
    # store losses per epoch
    trn_losses = [] # train loss
    val_losses = [] # validation loss
    best_val = 10000.0 # store the best val loss

    for epoch in range(args.epochs):

        # Train the Model
        model.train()  # Change model to 'train' mode
        start_tr = time.time()

        losses = []
        for b in range(0, (len(next(iter(wg.train))[0])), args.batch_size):
            features = next(iter(wg.train))[0].numpy()
            # tf_tensor = tf.convert_to_tensor(np_tensor)
            target = next(iter(wg.train))[1].numpy()
            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32) 

            if args.gpu: # if gpu
                X_batch = X_batch.cuda() # convert data to gpu
                y_batch = y_batch.cuda() # convert target to gpu

            ### Forward + Backward + Optimize ###
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(X_batch.float())

            # Calculate Loss
            # loss = model.sample_elbo(inputs=X_batch,
            #                labels=y_batch,
            #                criterion=torch.nn.L1Loss(),
            #                sample_nbr=3)
            loss = model.compute_loss(X_batch, y_batch, outputs)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            # store loss within one epoch
            losses.append(loss.data.item())

        # store train loss
        trn_loss = np.mean(losses) # mean loss of one epoch
        trn_losses.append(trn_loss) # store loss per epoch
        time_elapsed = time.time() - start_tr # record time
        print(
            "Epoch [%d/%d], Train Loss: %.4f, Training Time (s): %d"
            % (epoch + 1, args.epochs, trn_loss, time_elapsed)
        ) # print log        ### Validate the model ###
        start_val = time.time() # record time
        val_loss = validation(args, model, wg) # run validation
        val_losses.append(val_loss) # store val loss
        if val_loss < best_val: # compare if the current acc is better than the best acc obtained so far
            best_at = epoch + 1 # record best acc occurs epoch
            best_val = val_loss # record best acc
        # Print Loss
        time_elapsed = time.time() - start_val # record time
        print(
              "Epoch [%d/%d], Val Loss: %.4f,  Validation time (s): %d"
              % (epoch + 1, args.epochs, val_loss, time_elapsed)
          ) # print log

    print("===== Evaluation =====")
    # Calculate Test Accuracy         
    y_pred_test, y_real_test = test(args, rbscaler,  model, wg, full_model=args.full_model) 
    
    # forecast
    y_pred, y_real = fetch_all_pred_real2(args, model, wg, rbscaler)
    return y_pred_test, y_real_test, y_pred, y_real 
    
def validation(args, model, wg):
      model.train()   # Change model to 'train' mode 
      with torch.no_grad():
          # Iterate through validation dataset
          losses = []
          for b in range(0, (len(next(iter(wg.val))[0])), args.batch_size):
                  features = next(iter(wg.val))[0].numpy()
                  # tf_tensor = tf.convert_to_tensor(np_tensor)
                  target = next(iter(wg.val))[1].numpy()
                  X_batch = torch.tensor(features,dtype=torch.float32)    
                  y_batch = torch.tensor(target,dtype=torch.float32) 
                  if args.gpu: # if gpu
                      X_batch = X_batch.cuda() # convert data to gpu
                      y_batch = y_batch.cuda() # convert target to gpu

                  # # Load images to a Torch Variable
                  X_batch = X_batch.requires_grad_(False)
                  # Forward pass only to get logits/output
                  outputs = model(X_batch.float())
                  loss = model.compute_loss(X_batch, y_batch, outputs)
                  losses.append(loss.data.item()) # store loss
      val_loss = np.mean(losses) # mean loss of one epochh
      return val_loss
      
def test(args, rbscaler, model, wg, full_model=False): 
    if full_model == True:
      model.eval()    
    else:
      model.train()
    losses = []    
    with torch.no_grad():
      Xtest =  next(iter(wg.test))[0].numpy()
      ytest =  next(iter(wg.test))[1].numpy()
      if args.gpu: # if gpu
          Xtest =  torch.from_numpy(Xtest).cuda() # convert data to gpu
          ytest = torch.from_numpy(ytest).cuda() # convert target to gpu
      y_pred = model(Xtest.float())
      med, qrg = rbscaler.get_fitted()
      target_median, qrange = med.values[-1], qrg[-1]
      y_pred = y_pred.cpu().detach().numpy()*qrange+target_median
      y_real = np.array([y*qrange+target_median for y in ytest.cpu().detach().numpy()]).reshape(y_pred.shape)
      print(
        'MAE', list(MAE(y_pred, y_real)[0]), ' Mean MAE', MAE(y_pred, y_real)[1],
        '\nRMSE', list(RMSE(y_pred, y_real)[0]), 'Mean RMSE', RMSE(y_pred, y_real)[1],
        '\nMAPE', list(MAPE(y_pred, y_real)), 'Mean MAPE', MAPE(y_pred, y_real)[1],
        )
      return y_pred, y_real