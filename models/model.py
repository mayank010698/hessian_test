import torch
import copy
from .utils import get_model
from .utils import get_gradients
from .utils import find_cosine_dist


class FedModel:
    """
        Federated model.
    """

    def __init__(self, params, device=None):
        self.params = params
        self.device = device
        self.model, self.optimizer = get_model(params, device)
        self.model = self.model.to(device)
        self.model_size = self.compute_model_size()  # bit

    def compute_model_size(self):
        """
            Assume torch.FloatTensor --> 32 bit
        """
        tot_params = 0
        for param in self.model.parameters():
            tot_params += param.numel()
        return tot_params * 32

    def inference(self, x_input):
        with torch.no_grad():
            self.model.eval()
            return self.model(x_input)

    def compute_delta(self, data_loader):
        """
            In case of SignSGD or Fedavg, compute the gradient of the local model.
        """

        delta = dict()
        xt = dict()
        for k, v in self.model.named_parameters():
            delta[k] = torch.zeros_like(v)
            xt[k] = copy.deepcopy(v)

        # Update local model
        loss, cosine_sim_dist = self.perform_local_epochs(data_loader)
        for k, v in self.model.named_parameters():
            delta[k] = v - xt[k]

        return loss, delta, cosine_sim_dist


    def perform_local_epochs(self, data_loader):
        """
            Compute local epochs, the training stategies depends on the adopted model.
        """
        train_loss = None

        for epoch in range(self.params.get('model').get('local_epochs')):
            running_loss = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0
            avg_cosine_sim = 0
            num_iters = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                # print(torch.cuda.is_available())
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(train_x)

                loss = criterion(y_pred, train_y)
                running_loss += loss.item()
                pred_x, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                if epoch == 0 and batch_idx == 0:
                    prev_grads = get_gradients(self.model)
                else:
                    cosine_sim_dist = find_cosine_dist(self.model, prev_grads)
                    avg_cosine_sim += cosine_sim_dist
                    num_iters += 1.0
                    prev_grads = get_gradients(self.model)
                self.optimizer.step()
            train_loss = running_loss / total
            if self.params.get('simulation').get('verbose'):
                accuracy = correct / total
                print("Epoch {}: train loss {}  -  Accuracy {}".format(epoch + 1, train_loss, accuracy))
        avg_cosine_sim = avg_cosine_sim / num_iters
        print("Average cosine similarity:{}" .format(avg_cosine_sim))
        return train_loss, avg_cosine_sim

    def set_weights(self, w):
        self.model.load_state_dict(
            copy.deepcopy(w)
        )

    def get_weights(self):
        return self.model.state_dict()

    def save(self, folderpath):
        torch.save(self.model.state_dict(), folderpath.joinpath("local_model"))

    def load(self, folderpath):
        self.model.load_state_dict(torch.load(folderpath.joinpath("local_model"),
                                              map_location=torch.device('cpu')))
