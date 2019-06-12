import numpy as np
import torch
import torch.nn as nn
from utils import vectorization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        if args.action == 'train':
            args.b == 0
        self.args = args

        self.NOUT = 1 + self.args.M * 6  # end_of_stroke, num_of_gaussian * (pi + 2 * (mu + sigma) + rho)
        self.fc_output = torch.nn.Linear(args.rnn_state_size, self.NOUT)

        #if args.mode == 'predict':
        self.stacked_cell = torch.nn.LSTM(input_size=3, hidden_size=args.rnn_state_size, num_layers=2, batch_first=True)
        #else: # synthesis
        self.rnn_cell1 = nn.LSTMCell(input_size=3 + args.c_dimension, hidden_size=args.rnn_state_size)
        self.rnn_cell2 = nn.LSTMCell(input_size=3 + args.c_dimension + args.rnn_state_size, hidden_size=args.rnn_state_size)
        self.h2k = nn.Linear(args.rnn_state_size, args.K * 3)
        self.u = torch.arange(args.U).float().unsqueeze(0).repeat(args.K, 1) # (args.K, args.U)
        self.u = self.u.unsqueeze(0).repeat(args.batch_size, 1, 1).to(device) # (B, args.K, args.U)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)


    def _bivariate_gaussian(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        z = torch.pow((x1 - mu1) / sigma1, 2) + torch.pow((x2 - mu2) / sigma2, 2) \
            - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
        return torch.exp(-z / (2 * (1 - torch.pow(rho, 2)))) / \
               (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - torch.pow(rho, 2)))
   

    def fit(self, x, y, c_vec=None):
        '''
            x: (batch_size, args.T, 3) # args.T=300 if train else 1, (batch_size, T, 3)
            y: (batch_size, args.T, 3)
            c_vec: (batch_size, 12, 54)
        '''
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        if self.args.mode == 'predict':
            output_list, final_state = self.stacked_cell(x, None)

        else: # synthesis
            w = torch.zeros(self.args.batch_size, self.args.c_dimension).to(device)
            kappa_prev = torch.zeros([self.args.batch_size, self.args.K, 1]).to(device)
            cell1_state, cell2_state = None, None

            output_list = torch.zeros(self.args.batch_size, self.args.T, self.args.rnn_state_size).to(device)
            for t in range(self.args.T):
                cell1_state = self.rnn_cell1(torch.cat([x[:,t,:], w], 1), cell1_state) # input: (B, 3 + c_dimension)
                k_gaussian = self.h2k(cell1_state[0]) # (B, K * 3)

                alpha_hat, beta_hat, kappa_hat = torch.split(k_gaussian, self.args.K, dim=1) # (B, K)

                alpha = torch.exp(alpha_hat).unsqueeze(2) # (B, K, 1)
                beta = torch.exp(beta_hat).unsqueeze(2) # (B, K, 1)

                self.kappa = kappa_prev + torch.exp(kappa_hat).unsqueeze(2) # (B, K, 1)
                kappa_prev = self.kappa

                # self.u: (B, K, U). self.kappa: (B, K, 1)
                self.phi = torch.sum(torch.exp(torch.pow(-self.u + self.kappa, 2) * (-beta)) * alpha, 1, keepdim=True) # (B, 1, U)

                # c_vec: (B, U, c_dimension)
                w = torch.squeeze(torch.matmul(self.phi, torch.Tensor(c_vec).to(device)), 1) # (B, c_dimension), torch.matmul can execute batch_mm.

                cell2_state = self.rnn_cell2(torch.cat([x[:,t,:], cell1_state[0], w], 1), cell2_state)

                #output_list.append(cell2_state[0])
                output_list[:, t,:] = cell2_state[0]

        output = self.fc_output(output_list.reshape(-1, self.args.rnn_state_size)) # (batch_size * args.T, self.NOUT=121)
        y1, y2, y_end_of_stroke = torch.unbind(y.view(-1, 3), dim=1) # (batch_size * args.T, )

        end_of_stroke = 1 / (1 + torch.exp(output[:, 0])) # (batch_size * args.T,) 
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(output[:, 1:], self.args.M, 1)

        pi = pi_hat.softmax(1)

        sigma1 = torch.exp(sigma1_hat - self.args.b)
        sigma2 = torch.exp(sigma2_hat - self.args.b)
        rho = torch.tanh(rho_hat)
        gaussian = pi * self._bivariate_gaussian(
            y1.unsqueeze(1).repeat(1, self.args.M), y2.unsqueeze(1).repeat(1, self.args.M),
            mu1, mu2, sigma1, sigma2, rho
        )
        eps = 1e-20
        loss_gaussian = torch.sum(-torch.log(torch.sum(gaussian, 1) + eps))
        loss_bernoulli = torch.sum(
            -torch.log((end_of_stroke + eps) * y_end_of_stroke # e_t * (x_{t+1})_3 + (1 - e_t) * (1 - (x_{t+1})_3)
                    + (1 - end_of_stroke + eps) * (1 - y_end_of_stroke))
        )

        self.loss = (loss_gaussian + loss_bernoulli) / (self.args.batch_size * self.args.T)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



    def sample(self, length, s=None):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1 # The first point state is set to be 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]


        if self.args.mode == 'predict':
            final_state = None
        else:
            cell1_state, cell2_state = None, None
            w = torch.zeros(1, self.args.c_dimension).to(device)
            kappa_prev = torch.zeros(1, self.args.K, 1).to(device)
            
        output_list = torch.zeros(1, length, self.args.rnn_state_size).to(device)
        for t in range(length - 1):
            if self.args.mode == 'predict':
                output_list, final_state = self.stacked_cell(torch.Tensor(x).to(device), final_state) # !!! The final state argument is important because the PyTorch LSTM would initialize its states otherwise. Hence, we suggest that we should alwayes call LSTM with its initial states. None represents the empty states.
            else:
                x = torch.Tensor(x).to(device)

                cell1_state = self.rnn_cell1(torch.cat([x[:,0,:], w], 1), cell1_state) # input: (B, 3 + c_dimension)
                k_gaussian = self.h2k(cell1_state[0]) # (B, K * 3)

                alpha_hat, beta_hat, kappa_hat = torch.split(k_gaussian, self.args.K, dim=1) # (B, K)

                alpha = torch.exp(alpha_hat).unsqueeze(2) # (B, K, 1)
                beta = torch.exp(beta_hat).unsqueeze(2) # (B, K, 1)

                self.kappa = kappa_prev + torch.exp(kappa_hat).unsqueeze(2) # (B, K, 1)
                kappa_prev = self.kappa

                self.phi = torch.sum(torch.exp(torch.pow(-self.u + self.kappa, 2) * (-beta)) * alpha, 1, keepdim=True) # (B, K, 1)

                w = torch.squeeze(torch.matmul(self.phi, torch.Tensor([s]).to(device)), 1) # torch.matmul can execute batch_mm.

                cell2_state = self.rnn_cell2(torch.cat([x[:,0,:], cell1_state[0], w], 1), cell2_state)

                #output_list.append(cell2_state[0])
                output_list[:, t,:] = cell2_state[0]


            output = self.fc_output(output_list.reshape(-1, self.args.rnn_state_size)) # (1, NOUT:121)
            end_of_stroke = 1 / (1 + torch.exp(output[:, 0])) # (1, )
            pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(output[:, 1:], self.args.M, 1)
            pi_exp = torch.exp(pi_hat * (1 + self.args.b)) # args.b=3
            pi_exp_sum = torch.sum(pi_exp, 1)
            pi = pi_exp / (pi_exp_sum.unsqueeze(1).repeat(1, self.args.M))
            sigma1 = torch.exp(sigma1_hat - self.args.b)
            sigma2 = torch.exp(sigma2_hat - self.args.b)
            rho = torch.tanh(rho_hat)
            end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho = end_of_stroke.cpu().detach().numpy(), pi.cpu().detach().numpy(), mu1.cpu().detach().numpy(), mu2.cpu().detach().numpy(), sigma1.cpu().detach().numpy(), sigma2.cpu().detach().numpy(), rho.cpu().detach().numpy()

            x = np.zeros([1, 1, 3], np.float32)
#            if self.args.sample_random == True:
#                r = np.random.rand()
#                accu = 0
#                for m in range(self.args.M):
#                    accu += pi[0, m]
#                    if accu > r:
#                        x[0, 0, 0:2] = np.random.multivariate_normal(
#                            [mu1[0, m], mu2[0, m]],
#                            [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
#                             [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
#                        )
#                        break
#            else:
            for m in range(self.args.M):
                x[0, 0, 0:2] += pi[0, m] * np.random.multivariate_normal(
                        [mu1[0, m], mu2[0, m]],
                        [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
                         [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
                    )
            e = np.random.rand() # bernouli
            if e < end_of_stroke:
                x[0, 0, 2] = 1
            else:
                x[0, 0, 2] = 0
            strokes[t + 1, :] = x[0, 0, :]
        return strokes
