import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_random', type=bool, default=False,
                    help='apply random-control sample or not')

parser.add_argument('--text', type=str, default='a quick brown fox jumps over the lazy dog', help='')
parser.add_argument('--mode', type=str, default='synthesis',
                    help='')
parser.add_argument('--model_path', type=str, default='./data/pkl/model.pkl',
                    help='the model parameters file to load')
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--chars', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
                    help='chars')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='num of epochs')
parser.add_argument('--T', type=int, default=300,
                    help='RNN sequence length')
parser.add_argument('--points_per_char', type=int, default=25,
                    help='points per char (appr.)')
parser.add_argument('--rnn_state_size', type=int, default=400,
                    help='RNN hidden state size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='num of RNN stack layers')
parser.add_argument('--M', type=int, default=20,
                    help='num of mixture bivariate gaussian')
parser.add_argument('--K', type=int, default=5,
                    help='num of mixture bivariate gaussian (for synthesis)')
parser.add_argument('--data_scale', type=float, default=20,
                    help='factor to scale raw data down by')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--b', type=float, default=3.0,
                    help='biased sampling')
args = parser.parse_args()
