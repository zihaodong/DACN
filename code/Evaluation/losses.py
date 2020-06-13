import torch
import torch.nn.functional as F
import numpy as np
import Utils.utils as utils
import Evaluation.focal_loss as focal
#from Evaluation import losses
from torch.autograd import Variable
from torch import nn
import math
from sklearn.utils.extmath import cartesian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """

    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss

def fp_edge_loss(gt_edges, edge_logits):
    """
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """
    edges_shape = gt_edges.size()
    gt_edges = gt_edges.view(edges_shape[0], -1)
    #print(edge_logits.shape)
    #print(gt_edges.shape)
    
    #edge_logits = edge_logits.view(gt_edges.size()[0],28,28)
    
    #loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)
    loss = class_balanced_cross_entropy_loss(edge_logits, gt_edges)
    #print(edge_logits.shape)
    #print(gt_edges.shape)
    #focalloss = focal.FocalLoss()
    #loss = focalloss(edge_logits, gt_edges)
    return torch.mean(loss)

def fp_vertex_loss(gt_verts, vertex_logits):
    """
    Vertex loss in the first point network
    
    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """
    verts_shape = gt_verts.size()
    gt_verts = gt_verts.view(verts_shape[0], -1)
    #gt_verts = gt_verts.long()
    #loss = focal.FocalLoss(gamma=0)(Variable(vertex_logits.cuda()),Variable(gt_verts.cuda()))
    loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)
    #loss = class_balanced_cross_entropy_loss(vertex_logits, gt_verts)
    return torch.mean(loss)


def poly_mathcing_loss(pnum, pred, gt, loss_type="L2"):

    batch_size = pred.size()[0]
    
    pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
    for b in range(batch_size):
        for i in range(pnum):
            pidx = (np.arange(pnum) + i) % pnum
            pidxall[b, i] = pidx

    pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)
    # import ipdb;
    # ipdb.set_trace()
    feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), gt.size(2)).detach()
    gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

    pred_expand = pred.unsqueeze(1)
    dis = pred_expand - gt_expand
    #distances = torch.sum(dis**2, -1).sqrt()

    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        dis = torch.abs(dis).sum(3).sum(2)
    
    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    #term_1 = torch.mean(torch.min(distances, 1)[0])
    #term_2 = torch.mean(torch.min(distances, 0)[0])

    #min_DIS = term_1+term_2
    min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
    min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
                            expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
    gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)
    return gt_right_order, torch.mean(min_dis)


def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

	
class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 return_2_terms=False,
                 device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = torch.tensor(self.all_img_locations,
                                              dtype=torch.get_default_dtype()).to(device)
        self.return_2_terms = return_2_terms


    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
         between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size of the original images.
                           B is batch size. The size must be in (height, width) format.
        :param orig_widths: List of the original width for each image in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance. 
        """

        #_assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) *\
                self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            eps = 1e-6
            alpha = 4

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + eps)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            d_div_p = torch.min((d_matrix + eps) /
                                (p_replicated**alpha + eps / self.max_dist), 0)[0]
            d_div_p = torch.clamp(d_div_p, 0, self.max_dist)
            term_2 = torch.mean(d_div_p, 0)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()
        return res

'''
def gauss_match(pred, gt)

    pred_expand = 
    gt_expand = 
    dis = pred_expand - gt_expand
    dis = (dis ** 2).sum(3).sqrt().sum(2)
    return torch.mean(dis)
'''
