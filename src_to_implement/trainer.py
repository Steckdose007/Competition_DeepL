import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # Forward pass
        outputs = self._model(x)
        loss = self._crit(outputs, y)

        # Backward and optimize
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        del x, y, outputs
        t.cuda.empty_cache()
        return loss

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        pred = self._model(x)
        loss = self._crit(pred, y)
        del x, y
        return loss, pred

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        losses = 0
        for i, (images, labels) in enumerate(tqdm(self._train_dl)):
            # Move tensors to the configured device
            images = images.cuda()
            labels = labels.cuda()
            losses += self.train_step(images, labels.float())
        losses = losses / i
        return losses

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            correct = 0
            total = 0
            losses = 0
             # iterate through the validation set
            for i, (images, labels) in enumerate(self._val_test_dl):
                # transfer the batch to the gpu if given

                images = images.cuda()
                labels = labels.cuda()
        # perform a validation step
                loss, pred = self.val_test_step(images, labels.float())
                # print(pred[0])
                # print(pred)
                # print(pred[0,0])
        # save the predictions and the labels for each batch
                losses += loss
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
                #print("before:",pred)
                pred = (pred>0.5).float()
                #print("after:", pred)
                #print("label",labels)
                #score = f1_score(labels.cpu(), pred.cpu(), average=None)
                del images, labels, pred
            losses = losses / i
        # return the loss and print the calculated metrics
            #print("Class 1 score:", score[0])
            #print("Class 2 score:", score[1])
            #print("Average score:", (score[0] + score[1]) / 2)
            return losses

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_loss = []
        val_loss = []
        epoch = 0
        has_not_improved = 0
        while True:
            # stop by epoch number
            if (epoch == epochs):
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            loss = self.train_epoch()
            train_loss.append(loss)
            # append the losses to the respective lists
            valloss = self.val_test()
            val_loss.append(valloss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if (loss < train_loss[-1]):
                self.save_checkpoint()
                has_not_improved = 0
            else:
                has_not_improved += 1
                if (has_not_improved == self._early_stopping_patience):
                    break
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            epoch += 1
            # return the losses for both training and validation
            print('Epoch [{}/{}], Train_loss: {:.4f}, Val_los: {:.4f}'.format(epoch, epochs, loss, valloss))
        return train_loss, val_loss
