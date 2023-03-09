from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
from tbsim.utils.geometry_utils import batch_nd_transform_points


def bilinear_interpolate(img, x, y, floattype=torch.float, flip_y=False):
    """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
    Args:
        img (torch.Tensor): Tensor of size cxwxh. Usually one channel of feature layer
        x (torch.Tensor): Float dtype, x axis location for sampling
        y (torch.Tensor): Float dtype, y axis location for sampling

    Returns:
        torch.Tensor: interpolated value
    """
    if flip_y:
        y = img.shape[-2]-1-y
    if img.device.type == "cuda":
        x0 = torch.floor(x).type(torch.cuda.LongTensor)
        y0 = torch.floor(y).type(torch.cuda.LongTensor)
    elif img.device.type == "cpu":
        x0 = torch.floor(x).type(torch.LongTensor)
        y0 = torch.floor(y).type(torch.LongTensor)
    else:
        raise ValueError("device not recognized")
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[-1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[-1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[-2] - 1)
    y1 = torch.clamp(y1, 0, img.shape[-2] - 1)

    Ia = img[..., y0, x0]
    Ib = img[..., y1, x0]
    Ic = img[..., y0, x1]
    Id = img[..., y1, x1]

    step = (x1.type(floattype) - x0.type(floattype)) * (
        y1.type(floattype) - y0.type(floattype)
    )
    step = torch.clamp(step, 1e-3, 2)
    norm_const = 1 / step

    wa = (x1.type(floattype) - x) * (y1.type(floattype) - y) * norm_const
    wb = (x1.type(floattype) - x) * (y - y0.type(floattype)) * norm_const
    wc = (x - x0.type(floattype)) * (y1.type(floattype) - y) * norm_const
    wd = (x - x0.type(floattype)) * (y - y0.type(floattype)) * norm_const
    return (
        Ia * wa.unsqueeze(0)
        + Ib * wb.unsqueeze(0)
        + Ic * wc.unsqueeze(0)
        + Id * wd.unsqueeze(0)
    )


def ROI_align(features, ROI, outdim):
    """Given feature layers and proposals return bilinear interpolated
    points in feature layer

    Args:
        features (torch.Tensor): Tensor of shape channels x width x height
        proposal (list of torch.Tensor): x0,y0,W1,W2,H1,H2,psi
    """

    bs, num_channels, h, w = features.shape

    xg = (
        torch.cat(
            (
                torch.arange(0, outdim).view(-1, 1) - (outdim - 1) / 2,
                torch.zeros([outdim, 1]),
            ),
            dim=-1,
        )
        / outdim
    )
    yg = (
        torch.cat(
            (
                torch.zeros([outdim, 1]),
                torch.arange(0, outdim).view(-1, 1) - (outdim - 1) / 2,
            ),
            dim=-1,
        )
        / outdim
    )
    gg = xg.view(1, -1, 2) + yg.view(-1, 1, 2)
    gg = gg.to(features.device)
    res = list()
    for i in range(bs):
        if ROI[i] is not None:
            W1 = ROI[i][..., 2:3]
            W2 = ROI[i][..., 3:4]
            H1 = ROI[i][..., 4:5]
            H2 = ROI[i][..., 5:6]
            psi = ROI[i][..., 6:]
            WH = torch.cat((W1 + W2, H1 + H2), dim=-1)
            offset = torch.cat(((W1 - W2) / 2, (H1 - H2) / 2), dim=-1)
            s = torch.sin(psi).unsqueeze(-1)
            c = torch.cos(psi).unsqueeze(-1)
            rotM = torch.cat(
                (torch.cat((c, -s), dim=-1), torch.cat((s, c), dim=-1)), dim=-2
            )
            ggi = gg * WH[..., None, None, :] - offset[..., None, None, :]
            ggi = ggi @ rotM[..., None, :, :] + ROI[i][..., None, None, 0:2]

            x_sample = ggi[..., 0].flatten()
            y_sample = ggi[..., 1].flatten()
            res.append(
                bilinear_interpolate(features[i], x_sample, y_sample).view(
                    ggi.shape[0], num_channels, *ggi.shape[1:-1]
                )
            )
        else:
            res.append(None)

    return res


def generate_ROIs(
        pos,
        yaw,
        raster_from_agent,
        mask,
        patch_size,
        mode="last",
):
    """
    This version generates ROI for all agents only at most recent time step unless specified otherwise
    """
    if mode == "all":
        bs = pos.shape[0]
        yaw = yaw.type(torch.float)
        Mat = raster_from_agent.view(-1, 1, 1, 3, 3).type(torch.float)
        raster_xy = batch_nd_transform_points(pos, Mat)
        raster_mult = torch.linalg.norm(
            raster_from_agent[0, 0, 0:2], dim=[-1]).item()
        patch_size = patch_size.type(torch.float)
        patch_size *= raster_mult
        ROI = [None] * bs
        index = [None] * bs
        for i in range(bs):
            ii, jj = torch.where(mask[i])
            index[i] = (ii, jj)
            if patch_size.ndim == 1:
                patches_size = patch_size.repeat(ii.shape[0], 1)
            else:
                sizes = patch_size[i, ii]
                patches_size = torch.cat(
                    (
                        sizes[:, 0:1] * 0.5,
                        sizes[:, 0:1] * 0.5,
                        sizes[:, 1:2] * 0.5,
                        sizes[:, 1:2] * 0.5,
                    ),
                    dim=-1,
                )
            ROI[i] = torch.cat(
                (
                    raster_xy[i, ii, jj],
                    patches_size,
                    yaw[i, ii, jj],
                ),
                dim=-1,
            ).to(pos.device)
        return ROI, index
    elif mode == "last":
        num = torch.arange(0, mask.shape[2]).view(1, 1, -1).to(mask.device)
        nummask = num * mask
        last_idx, _ = torch.max(nummask, dim=2)
        bs = pos.shape[0]
        Mat = raster_from_agent.view(-1, 1, 1, 3, 3).type(torch.float)
        raster_xy = batch_nd_transform_points(pos, Mat)
        raster_mult = torch.linalg.norm(
            raster_from_agent[0, 0, 0:2], dim=[-1]).item()
        patch_size = patch_size.type(torch.float)
        patch_size *= raster_mult
        agent_mask = mask.any(dim=2)
        ROI = [None] * bs
        index = [None] * bs
        for i in range(bs):
            ii = torch.where(agent_mask[i])[0]
            index[i] = ii
            if patch_size.ndim == 1:
                patches_size = patch_size.repeat(ii.shape[0], 1)
            else:
                sizes = patch_size[i, ii]
                patches_size = torch.cat(
                    (
                        sizes[:, 0:1] * 0.5,
                        sizes[:, 0:1] * 0.5,
                        sizes[:, 1:2] * 0.5,
                        sizes[:, 1:2] * 0.5,
                    ),
                    dim=-1,
                )
            ROI[i] = torch.cat(
                (
                    raster_xy[i, ii, last_idx[i, ii]],
                    patches_size,
                    yaw[i, ii, last_idx[i, ii]],
                ),
                dim=-1,
            )
        return ROI, index
    else:
        raise ValueError("mode must be 'all' or 'last'")


def Indexing_ROI_result(CNN_out, index, emb_size):
    """put the lists of ROI align result into embedding tensor with the help of index"""
    bs = len(CNN_out)
    map_emb = torch.zeros(emb_size).to(CNN_out[0].device)
    if map_emb.ndim == 3:
        for i in range(bs):
            map_emb[i, index[i]] = CNN_out[i]
    elif map_emb.ndim == 4:
        for i in range(bs):
            ii, jj = index[i]
            map_emb[i, ii, jj] = CNN_out[i]
    else:
        raise ValueError("wrong dimension for the map embedding!")

    return map_emb
