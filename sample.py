import os
import torch
import numpy as np
import model as m
from utils import *
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
#s = 'a quick brown fox jumps over the lazy dog'
s = args.text
# str = 'aaaaabbbbbccccc'
args.U = len(s)
args.c_dimension = len(data_loader.chars) + 1
args.T = 1
args.batch_size = 1
args.action = 'sample'

model = m.Model(args).to(device)

if args.model_path and os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    if args.mode == 'predict':
        strokes = model.sample(800)
    else:
        vec = vectorization(s, data_loader.char_to_indices)
        strokes = model.sample(len(s) * args.points_per_char, s=vec)
        
    #print(strokes)
    draw_strokes_random_color(strokes, factor=0.1, svg_filename='images/sample.normal.svg')
