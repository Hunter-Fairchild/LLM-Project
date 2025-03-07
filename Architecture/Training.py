import torch
import datetime
from Architecture.Loss_Functions import calc_loss_batch, calc_loss_loader
from Architecture.Generation import generate_and_print


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Optimization routine for training GPT Models"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    start = datetime.datetime.now()
    
    for epoch in range(num_epochs):             # loop over number of epochs (passes over the training data)
        model.train()                           # put model in training mode

        for input_batch, target_batch in train_loader:                        # loop over the training data       (randomized)
            optimizer.zero_grad()                                             # sets all gradients to zero        (zero'ing previous samples)
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # evaluate (sample of) function
            loss.backward()                                                   # compute (sample of) gradient
            optimizer.step()                                                  # perform optimization step         (based on current sample)


            tokens_seen += input_batch.numel()                         # bookkeeping information
            global_step += 1

            if global_step % eval_freq == 0:                           # debug evaluation of loss and printing
                train_loss, val_loss = evaluate_model(                     # compute losses
                    model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)                            # store loss
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)                      # store tokens seen
                
                end = datetime.datetime.now() - start
                end = end - datetime.timedelta(microseconds=end.microseconds)
                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "           # debug print
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}, "
                    f"Time {end}"
                )

        generate_and_print(model, tokenizer, device, start_context)        # output from the model for each epoch

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """computes the loss for both the training and validation sets"""

    model.eval() # set to evaluation mode

    with torch.no_grad(): # compute losses
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train() # set to training mode

    return train_loss, val_loss