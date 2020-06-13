import os
import numpy as np
import wget
import argparse
import base64
import json
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch

from Utils import utils
from Models.GNN import poly_gnn
from torch.utils.data import DataLoader
from DataProvider import cityscapes_for_tool
from active_spline import ActiveSplineTorch

from skimage.segmentation import active_contour
from skimage.filters import gaussian
from contour import activeContour

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True)
    parser.add_argument('--reload', required=True)
    parser.add_argument('--image_dir', default='static/images/')
    parser.add_argument('--port', type=int, default=5001)

    args = parser.parse_args()
    return args

def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'
    data_loader = DataProvider(split='val', opts=opts['train_val'], mode='tool')
    return data_loader

class Tool(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.image_dir = args.image_dir
        self.data_loader = get_data_loaders(self.opts['dataset'], cityscapes_for_tool.DataProvider)
        self.model = poly_gnn.PolyGNN(state_dim=self.opts['state_dim'],
                                      n_adj=self.opts['n_adj'],
                                      cnn_feature_grids=self.opts['cnn_feature_grids'],
                                      coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                      get_point_annotation=self.opts['get_point_annotation'],
                                      ).to(device) 
        self.model.reload(args.reload, strict=False)
        self.spline = ActiveSplineTorch(self.opts['cp_num'], self.opts[u'p_num'], device=device)
        self.spline1 = ActiveSplineTorch(127, self.opts[u'p_num'], device=device)

    def get_grid_size(self, run_ggnn=True):
        grid_size = self.model.grid_size
        return grid_size

    def annotation(self, instance, run_ggnn=False):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['fwd_poly'], 0)
            img = torch.from_numpy(img).to(device)
            poly = torch.from_numpy(poly)#.to(device)
            output = self.model(img,poly)
            polys = output['pred_polys'][-1].cpu().numpy()
        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

    def fixing(self, instance, run_ggnn=False):
        with torch.no_grad():
            #print(instance)
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['fwd_poly'], 0) 
            img = torch.from_numpy(img).to(device)
            poly = torch.from_numpy(poly).to(device)    
            #Modify DACN model to GCN by zihao Dong // Global to local
            output = self.model(img,poly)
            polys = output['pred_polys'][-1].cpu().numpy()  
            #polys = self.spline.sample_point(polys)
            #torch.cuda.synchronize()
            #polys = polys.cpu().numpy()
      
            #snakes experiment Zihao Dong
            #polys=active_contour(gaussian(img[-1], 3),poly[-1],alpha=0.015, beta=10, gamma=0.001)
        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output_fix(polys, instance,
            grid_size)

    def process_output_fix(self, polys, instance, grid_size):
        #polys = self.spline1.sample_point(torch.from_numpy(polys))
        #torch.cuda.synchronize()
        #poly = polys.cpu().numpy()[0]
        poly = polys[0]
        
        #del dzh
        #poly = utils.get_masked_poly(poly, grid_size)
        #poly = utils.class_to_xy(poly, grid_size)
        #poly = utils.poly0g_to_poly01(poly, grid_size)
        #polys = self.spline.sample_point(polys)
        #torch.cuda.synchronize()
        #poly = polys.cpu().numpy()[0]

        poly = poly * instance['patch_w']
        poly = poly + instance['starting_point']
        torch.cuda.empty_cache()
        return [poly.astype(np.int).tolist()]

    def process_output(self, polys, instance, grid_size):
        poly = polys[0]
        #poly = utils.get_masked_poly(poly, grid_size)
        #poly = utils.class_to_xy(poly, grid_size)
        #poly = utils.poly0g_to_poly01(poly, grid_size)
        
        #polys = self.spline.sample_point(polys)
        #torch.cuda.synchronize()
        #polys = polys.cpu().numpy()

        poly = poly * instance['patch_w']
        poly = poly + instance['starting_point']
        
        #poly = self.spline.sample_point(poly)
        #torch.cuda.empty_cache() 
        return [poly.astype(np.int).tolist()]

@app.route('/')
def index():
        return 'Hello World'

@app.route('/api/annotation', methods=['GET','POST'])
def generate_annotation():
    start = time.time()
    instance = request.json
    #print(instance)
    component = {}
    component['poly'] = np.array([[-1., -1.]])
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.annotation(instance)
    #pred_annotation[0] = pred_annotation[0][::2]
    print "Annotation time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/fix_poly', methods=['GET','POST'])
def fix_poly_request():
    start = time.time()
    instance = request.json
    #instance['poly'][10][0]=instance['poly'][10][0]+100
    # ddd = instance['bbox']
    index_fix = int(len(instance['poly']))
    component = {}
    component['poly'] = instance['poly']
    instance = tool.data_loader.prepare_component(instance, component)#.prepare_component_for_fixing(instance, component)
       
    pred_annotation = tool.fixing(instance, run_ggnn=False)
    # print(np.array(pred_annotation[0][index_fix:index_fix+5]))
    cc = np.array(pred_annotation[0][index_fix:index_fix+5])
    image_d = np.mean(instance['img'], axis=2)
    activeContour(image_d, cc) # pred_annotation[0][index_fix:index_fix+5])
    pred_annotation[0][index_fix:index_fix+5] = cc.tolist()

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/ggnn_poly', methods=['POST'])
def ggnn_poly_request():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = instance['poly']
    # instance = tool.data_loader.prepare_component(instance, component)
    instance = data_loader.prepare_component(instance, component)
    pred_annotation = tool.run_ggnn(instance)

    print "GGNN time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/upload_v3', methods=['POST'])
def upload_v3():
    instance = request.json
    url = instance['url']
    out_dir = tool.image_dir
    filename = wget.download(url, out=out_dir)
    response = jsonify(path=filename)
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/upload_v2', methods=['POST'])
def upload_v2():
    instance = request.json
    base64im = instance['image']
    idx = len(os.listdir(tool.image_dir))
    try:
        extension = base64im.split('/')[1].split(';')[0]
        t = base64im.split('/')[0].split(':')[1]
        assert t == 'image', 'Did not get image data!'
        
        base64im = base64im.split(',')[1]
        out_name = os.path.join(tool.image_dir, str(idx) + '.' + extension)

        with open(out_name, 'w') as f:
            f.write(base64.b64decode(base64im.encode()))

        response = jsonify(path=out_name)

    except Exception as e:
        print e
        response = jsonify(path='')

    response.headers['Access-Control-Allow-Headers'] = '*'
    return response
    
if __name__ == '__main__':
    args = get_args()
    global tool
    tool = Tool(args)
    app.run(host='0.0.0.0', threaded=True, port=args.port)
