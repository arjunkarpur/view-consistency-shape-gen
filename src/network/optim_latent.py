import torch
import config
import numpy as np
import torch.backends.cudnn as cudnn
import scipy.io as scio
from train import log_print

class RunningAverage():
    def __init__(self):
        self.sum = 0.0
        self.n = 0.0
        self.avg = 0.0

    def reset(self):
        self.sum = 0.0
        self.n = 0.0
        self.avg = 0.0

    def add(self, val, num=1):
        self.sum += float(val)
        self.n += num
        self.avg = float(self.sum) / float(self.n)

    def get(self):
        return self.avg

def fetch_Y(train_src_dataloader):
    Y_list = []
    Y_ind_map = {}
    ind_ctr = 0
    for data in train_src_dataloader:
        for i in xrange(len(data['id'])):
            id_ = str(data['id'][i])
            if id_ not in Y_ind_map:
                t = data['data'][i]
                dim = t.size(0)
                Y_list.append(t.view(1,1,dim,dim,dim))
                Y_ind_map[id_] = ind_ctr
                ind_ctr += 1
    return Y_list, Y_ind_map

def calc_avg(model_ae, model_im, train_target_dataloader, Y_list):

    model_ae.eval()
    model_im.eval()

    # Create tensor version of Y_list
    v_dim = Y_list[0].size(4)
    Y_var_ = torch.Tensor(len(Y_list), 1, v_dim, v_dim, v_dim)
    Y_var_ = torch.cat(Y_list)
    if config.GPU and torch.cuda.is_available():
        Y_var_ = Y_var_.cuda()
    Y_var = torch.autograd.Variable(Y_var_, requires_grad=False).float()

    # Calculate mean
    mean = RunningAverage()
    batch_sample = 999999999999 #TODO: set to proper sample number
    batch_count = 0
    for data in train_target_dataloader:
        if batch_count == batch_sample:
            break

        # Perform forward pass
        ims = data['im']
        if config.GPU and torch.cuda.is_available():
            ims = ims.cuda()
        ims = torch.autograd.Variable(ims).float()
        im_embed = model_im(ims)
        try:
            voxel_preds = model_ae.module._decode(im_embed)
        except:
            voxel_preds = model_ae._decode(im_embed)
        voxel_preds.detach()

        # Find min distance for each img in each batch
        loss_f = torch.nn.BCELoss(reduce=False).cuda()
        for i in xrange(ims.size(0)):
            curr_pred = voxel_preds[i].view(1,1,v_dim,v_dim,v_dim)
            curr_pred_list = [curr_pred for i in xrange(Y_var.size(0))]
            curr_pred_stack = torch.cat(curr_pred_list).cuda()

            l = loss_f(curr_pred_stack, Y_var)
            best_val = float('inf')
            for j in xrange(l.size(0)):
                val = torch.mean(l[j]).item()
                best_val = min(best_val, val)
            mean.add(best_val)
        batch_count += 1
    model_ae.train()
    model_im.train()
    return mean.get()

def calc_rho_density(M, Y_list, Y_ind_map, Y_im_counts, Y_im_counts_list, sigma_2):
    
    tmp = sigma_2
    sigma_2 = torch.Tensor(1)
    sigma_2[0] = tmp
    sigma_2 = torch.autograd.Variable(sigma_2.cuda())
    
    # Stack Y
    Y_var_ = torch.cat(Y_list).cuda()
    Y_var = torch.autograd.Variable(Y_var_, requires_grad=False).float()

    # Stack M
    v_dim = Y_var.size(4)
    M = M.view(1,1,v_dim,v_dim,v_dim)
    M_list = [M for i in xrange(len(Y_list))]
    M_stack = torch.cat(M_list).cuda()
    M_stack = torch.autograd.Variable(M_stack, requires_grad=False).float()

    # Losses
    loss_f = torch.nn.BCELoss(reduce=False).cuda()
    l = loss_f(M_stack, Y_var)

    means = torch.mean(torch.mean(torch.mean(torch.mean(l,dim=1), dim=1), dim=1), dim=1)
    vals = torch.exp((-1.0*means) / (2*sigma_2))
    total = torch.sum(torch.mul(vals, Y_im_counts_list))
    return total.item()

def voxelize_latents(M_list, thresh):
    for i in xrange(len(M_list)):
        M = M_list[i]
        M[M < thresh] = 0
        M[M >= thresh] = 1
        M_list[i] = M
    return M_list

def init_latents(model_ae, model_im, train_target_dataloader, Y_list, Y_ind_map, Y_im_counts, sigma_2):

    if config.VIEW_INIT_LATENTS is not None:
        data = scio.loadmat(config.VIEW_INIT_LATENTS)
        M_list_, M_ind_keys, M_ind_vals = data['M_list'], data['M_ind_keys'], data['M_ind_vals']
        M_list = []
        for m in M_list_:
            M = torch.from_numpy(m).cuda()
            M_list.append(M)
        M_ind_map = {}
        M_ind_keys = list(M_ind_keys)
        M_ind_vals = list(list(M_ind_vals)[0])
        for i in xrange(len(M_ind_keys)):
            M_ind_map[str(M_ind_keys[i])] = M_ind_vals[i]
        return M_list, M_ind_map

    model_ae.eval()
    model_im.eval()

    ind_ctr = 0
    M_im_counts = train_target_dataloader.dataset.im_counts
    M_rhos = [-float('inf') for i in xrange(len(M_im_counts))]
    M_list = [0 for i in xrange(len(M_im_counts))]
    M_ind_map = {}

    Y_im_counts_list = [None for i in xrange(len(Y_list))]
    for id_ in Y_im_counts:
        ind = Y_ind_map[id_]
        Y_im_counts_list[ind] = Y_im_counts[id_]
    Y_im_counts_list = (torch.from_numpy(np.array(Y_im_counts_list)).float()).cuda()

    count = 0
    print len(train_target_dataloader.dataset) / 32
    for data in train_target_dataloader:

        print count
        count += 1
        names = data['im_name']
        ims = data['im']
        ims = ims.cuda()
        ims = torch.autograd.Variable(ims).float()
        im_embed = model_im(ims)
        try:
            out_voxels = model_ae.module._decode(im_embed)
        except:
            out_voxels = model_ae._decode(im_embed)
        v_dim = out_voxels.size(3)

        for i in xrange(out_voxels.size(0)):
            name = names[i]
            id_ = name.split("_")[1]
            if id_ not in M_ind_map:
                M_ind_map[id_] = ind_ctr
                ind_ctr += 1
            ind = M_ind_map[id_]

            prop_M = out_voxels[i].view(1,1,v_dim,v_dim,v_dim)
            rho = calc_rho_density(prop_M, Y_list, Y_ind_map, Y_im_counts, Y_im_counts_list, sigma_2)
            if rho >= M_rhos[ind]:
                M_rhos[ind] = rho
                M_list[ind] = prop_M.data[0]

    init_latent_fp = "./init_latents.mat"
    M_list_ = []
    for m in M_list:
        M_list_.append(m.cpu().data.numpy())
    M_ind_keys = []
    M_ind_vals = []
    for id_ in M_ind_map:
        M_ind_keys.append(id_)
        M_ind_vals.append(M_ind_map[id_])
    scio.savemat(init_latent_fp, {"M_list": M_list_, "M_ind_keys": np.array(M_ind_keys), "M_ind_vals": np.array(M_ind_vals)})

    model_ae.train()
    model_im.train()
    return M_list, M_ind_map

def optimal_label(X, Z_list, Z_ind_map):
    best_id = None
    best_dist = float('inf')
    X = X.cuda().float()
    for id_ in Z_ind_map:
        Z = Z_list[Z_ind_map[id_]].cuda().float()
        dist = (torch.sum((X-Z)**2) / X.data.nelement())
        if dist < best_dist:
            best_id = id_
            best_dist = dist
    return best_id

def update_latents(model_ae, model_im, target_dataloader, M_list, M_ind_map, M_im_counts, Y_list, Y_ind_map, Y_im_counts, lambda_view, lambda_align):
    
    # Optimize each independently, given M
    model_ae.eval()
    model_im.eval()
    M_list_opt = [None for i in xrange(len(M_list))]
    N = len(M_list)
    src_cardinality = 0
    for id_ in Y_im_counts:
        src_cardinality += Y_im_counts[id_]

    # Preprocess images in dataloader, assign to 
    output_sums = [None for i in xrange(len(M_list))]
    for data in target_dataloader:
        ims, im_names = data['im'], data['im_name']
        ims = torch.autograd.Variable(ims.cuda(), requires_grad=False)
        im_embed = model_im(ims)
        try:
            out_voxels = model_ae.module._decode(im_embed)
        except:
            out_voxels = model_ae._decode(im_embed)

        for i in xrange(len(im_names)):
            im_name = im_names[i]
            id_ = im_name.split("_")[1]
            index = M_ind_map[id_]
            if output_sums[index] is None:
                output_sums[index] = torch.zeros(out_voxels[0].size())
            output_sums[index] += out_voxels[i].cpu().data
        del ims, im_names, im_embed, out_voxels

    # Update each latent var independently
    log_print("\t\t%i latent configurations" % len(M_ind_map))
    counter = 1
    for id_ in M_ind_map:

        # Init
        log_print("\t\tLatent %s (%i/%i)" % (id_, counter, len(M_ind_map)))
        index = M_ind_map[id_]
        M = M_list[index]

        # Align term one (closest image/label given M_i)
        closest_Y_id = optimal_label(M, Y_list, Y_ind_map)
        first_term_M = Y_list[Y_ind_map[closest_Y_id]]
        first_term_num = Y_im_counts[closest_Y_id]

        # Align term two (closest M_i given image/label)
        second_term_M = Y_list[Y_ind_map[closest_Y_id]]
        second_term_num = 1

        # View term
        third_term_M = output_sums[index] / M_im_counts[id_]
        third_term_num = M_im_counts[id_]

        # Save
        total = float(lambda_align)*float(first_term_num) + \
            float(lambda_align)*float(second_term_num) + \
            float(lambda_view)*float(third_term_num)
        weighted_first_M = ((float(lambda_align)*float(first_term_num))/float(total)) * first_term_M
        weighted_second_M = ((float(lambda_align)*float(second_term_num))/float(total)) * second_term_M
        weighted_third_M = ((float(lambda_view)*float(third_term_num))/float(total)) * third_term_M
        M_opt = (weighted_first_M.float() + weighted_second_M.float() + weighted_third_M.float()).cuda()
        M_list_opt[index] = M_opt.data
        counter += 1
        update_distance = (torch.sum((M_opt-M)**2) / M.data.nelement()).item()
        log_print("\t\t  Update distance: %f" % (update_distance))

    model_ae.train()
    model_im.train()
    return M_list_opt
