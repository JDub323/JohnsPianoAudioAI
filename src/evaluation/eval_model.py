# evaluation function: should be able to use the unseen test data to figure out error, confusion matrix, 
# and other important metrics, so I know how well my model is actually performing

def evaluate(config_path: str, checkpoint_path: str) -> None:
    # import the checkpoint 

    # load the model from the checkpoint, making sure to set to eval mode


    # load evaluation dataset
    # eval_data = PianoDataset(config['foo']['bar'], ...)
    # eval_loader = DataLoader(eval_data, batch_size=1)

    # evaluate 
    # scores = evaluate_transcription(model, eval_loader)

    # for all metrics and scores, print them out
    return

def evaluate_transcription(model, eval_loader):
    # model.eval() 
    # with torch.no_grad():
    #      for inputs, labels in val_loader:
    #         # have the model make predictions
    #         outputs = model(inputs)
    # 
    #         # calculate the loss function: outputs vs labels
    #         val_loss = criterion(outputs, labels)
    return
