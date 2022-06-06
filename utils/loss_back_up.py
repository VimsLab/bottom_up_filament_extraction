import torch
import torch.nn as nn
from config import cfg

def get_losses(ground_truth, outputs):

    mask_target, control_point_target, end_point_target, long_offset_target,\
         next_offset_target, prev_offset_target,control_short_offset_target, end_short_offset_target, \
         end_point_labels, control_point_labels, binary_targets = ground_truth

    mask_pred, control_point_pred, end_point_pred, long_offset_pred,\
         next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred, \
         end_point_tags, control_point_tags, binary_pred, refine_pred= outputs

    loss_mask = mask_loss(mask_target, mask_pred)
    loss_control_pt = kp_map_loss(control_point_target, control_point_pred)
    loss_end_pt = kp_map_loss(end_point_target, end_point_pred)

    loss_long_offset = long_offset_loss(long_offset_target, long_offset_pred)

    loss_next_offset = mid_offset_loss(next_offset_target, next_offset_pred)
    loss_prev_offset = mid_offset_loss(prev_offset_target, prev_offset_pred)

    loss_short_control_pt = short_offset_loss(control_short_offset_target, control_short_offset_pred)
    loss_short_end_pt = short_offset_loss(end_short_offset_target, end_short_offset_pred)

    loss_binary_targets = binary_targets_loss(binary_targets, binary_pred)
    # import pdb;pdb.set_trace()
    refine_loss = torch.ones(1).to(cfg.device)


    pull_loss = 0
    push_loss = 0
    # use refine to predict control_point tags
    control_embed_pred = refine_pred[:,0:3,:,:]
    # end_embed_pred = refine_pred[:,1:2,:,:]
    for control_point_label, control_point_tag in zip(control_point_labels, control_embed_pred):
        # import pdb;pdb.set_trace()
        pull, push = ae_loss(control_point_label, control_point_tag)
        pull_loss += pull
        push_loss += push
    pull_loss_control = pull_loss * 1.0
    push_loss_control = push_loss * 1.0

    # print (pull_loss_control)
    # print (push_loss_control)

    # for end_point_label, end_point_tag in zip(end_point_labels, end_embed_pred):

    #     pull, push = ae_loss(end_point_label, end_point_tag)
    #     pull_loss += pull
    #     push_loss += push
    pull_loss_end = pull_loss * 1.0
    push_loss_end = push_loss * 1.0

    losses = 0 * refine_loss + 0 * loss_binary_targets + 2 * (loss_control_pt + loss_end_pt)\
        + 0 * loss_mask + 0 * loss_long_offset + 0 * loss_next_offset + 0 * loss_prev_offset  \
        + 1 * loss_short_control_pt + 1 * loss_short_end_pt \
        + 1 * (loss_next_offset + loss_prev_offset) \
        + 2 * (pull_loss_end + push_loss_end)   \
        + 2 * (pull_loss_control + push_loss_control)
    # import pdb;pdb.set_trace()
    return losses, loss_mask, loss_control_pt, loss_long_offset, loss_next_offset, loss_short_control_pt,\
            pull_loss_end, push_loss_end, pull_loss_control, push_loss_control, loss_binary_targets, refine_loss

def mask_loss(mask_true, mask_pred):

    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss = criterion_bce(mask_pred, mask_true)
    loss = torch.mean(loss)
    return loss

def binary_targets_loss(mask_trues, mask_preds):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss_final = 0
    for mask_true, mask_pred in zip (mask_trues, mask_preds):

        loss = criterion_bce(mask_pred, mask_true)
        loss = torch.mean(loss)
        loss_final = loss_final + loss
    return loss_final

def kp_map_loss(kp_maps_true, kp_maps_pred):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss = criterion_bce(kp_maps_pred, kp_maps_true)
    loss = torch.mean(loss)
    return loss


def short_offset_loss(short_offset_true, short_offsets_pred):
    kp_maps_true = (short_offset_true!=0).type(torch.float32)
    short_offset_true = short_offset_true * kp_maps_true
    short_offsets_pred = short_offsets_pred * kp_maps_true


    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    loss = criterion_abs(short_offset_true, short_offsets_pred)/cfg.disc_radius * 1.0

    loss = loss / (torch.sum(kp_maps_true)+1)
    return loss


def mid_offset_loss(mid_offset_true, mid_offset_pred):
    kp_maps_true = (mid_offset_true!=0).type(torch.float32)
    mid_offset_true = mid_offset_true * kp_maps_true
    mid_offset_pred = mid_offset_pred * kp_maps_true

    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    loss = criterion_abs(mid_offset_true, mid_offset_pred)/cfg.disc_radius * 1.0

    loss = loss / (torch.sum(kp_maps_true)+1)
    return loss


def long_offset_loss(long_offset_true, long_offset_pred):
    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    seg_true = (long_offset_true!=0).type(torch.float32)

    long_offset_true = long_offset_true * seg_true
    long_offset_pred = long_offset_pred * seg_true


    loss = criterion_abs(long_offset_true, long_offset_pred)/cfg.disc_radius * 1.0
    loss = loss / (torch.sum(seg_true)+1)

    return loss


def segmentation_loss(seg_true, seg_pred):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss = criterion_bce(seg_true, seg_pred)
    loss = torch.mean(loss)
    return loss

def ae_loss(point_label, out_tag):
    # num of object
    import pdb;pdb.set_trace()
    out_tag = out_tag
    out_tag_flat = torch.flatten(out_tag.permute(1,2,0), end_dim=1)
    normed_out_tag = torch.nn.functional.normalize(out_tag_flat, p=2, dim=1)

    point_label_flat = torch.flatten(point_label) 
    label_ids = int(torch.unique(point_label).cpu().numpy())
    num = len(torch.unique(point_label_flat))
    # import pdb;pdb.set_trace()
    # mean of each instance
    # mean_ind = torch.zeros(num).to(cfg.device)
    mean_ind = []

    pull = 0
    for i, ind in enumerate(label_ids):
        if ind == 0:
            continue

        ind_tag = out_tag[point_label == ind]

        ind_tag_sum = ind_tag.sum()

        mean_ind.append(ind_tag_sum / (ind_tag.size()[0] + 1e-4))

        pull_of_this_instance_tmp = torch.pow(ind_tag - mean_ind[i], 2)

        import pdb;pdb.set_trace()
        pull_of_this_instance = pull_of_this_instance_tmp.sum() / (ind_tag.size()[0] + 1e-4)
        pull = pull + pull_of_this_instance

    pull = pull / (num + 1e-4)

    sum_dist = 0

    for i in range(num):
        for j in range(num):
            # import pdb;pdb.set_trace()
            if i != j :
                # import pdb;pdb.set_trace()
                dist = mean_ind[i] - mean_ind[j]
                dist = 1 - torch.abs(dist)
                dist = nn.functional.relu(dist, inplace=True)
                sum_dist = sum_dist + dist

    push = sum_dist / ((num - 1) * num + 1e-4)
    # import pdb;pdb.set_trace()
    # push = push.to(cfg.device)
    return pull, push



def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # n_loc, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

        _mean_sample = _pred_masked_sample.sum(
            0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means
