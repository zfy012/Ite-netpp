import torch
import torch.nn as nn
import torch.nn.functional as F


# Custom loss function
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true)

        smooth = 1e-5

        y_pred = torch.sigmoid(y_pred)
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true).float()

        intersection = (y_pred_f * y_true_f).sum(-1)
        denominator = (y_pred_f).sum(-1) + (y_true_f).sum(-1)

        dice = (2. * intersection + smooth) / (denominator + smooth)

        ldice = 1 - dice

        return 0.5 * bce + ldice


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights=None, smooth=1e-6):
        super(DiceLoss, self).__init__()

        self.num_classes = num_classes + 1  # with background

        if weights is None:
            weights = [1.0] * self.num_classes

        assert self.num_classes == len(weights)

        self.weights = weights
        self.smooth = smooth

    def flatten(self, tensor):
        # number of channels
        channel = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(channel, -1)

    def forward(self, y_pred, y_true):
        # (N, C, D, H, W) -> C = num_class + 1
        target = torch.zeros_like(y_pred)
        for index in range(self.num_classes):
            y_temp = torch.zeros_like(y_true)
            y_temp[y_true == index] = 1.0
            target[:, index, ...] = y_temp

        y_pred = self.flatten(y_pred)
        y_true = self.flatten(target).float()

        intersect = (y_pred * y_true).sum(-1)
        intersect = torch.tensor(self.weights).cuda() * intersect

        denominator = (y_pred).sum(-1) + (y_true).sum(-1)

        return 1. - torch.mean(
            2. * (intersect + self.smooth) / denominator + self.smooth)


def get_tp_fp_fn_tn(net_output, gt, axes=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_BCE_loss(nn.Module):
    def __init__(self, aggregate="sum"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss()
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result



