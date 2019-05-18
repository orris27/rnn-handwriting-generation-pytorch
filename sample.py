import os
import torch
import numpy as np
import model as m
from utils import *
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
str = 'a quick brown fox jumps over the lazy dog'
# str = 'aaaaabbbbbccccc'
args.U = len(str)
args.c_dimension = len(data_loader.chars) + 1
args.T = 1
args.batch_size = 1
args.action = 'sample'

model = m.Model(args).to(device)

if args.load_path and os.path.exists(args.load_path):
    model.load_state_dict(torch.load(args.load_path))
    model = model.eval()

    strokes = model.sample(800)
    print(strokes)
    draw_strokes_random_color(strokes, factor=0.1, svg_filename='sample' + '.normal.svg')
