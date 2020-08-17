# coding=utf-8
"""
Python module for implementing inner maximizers for robust adversarial training
(Table I in the paper)
"""
import torch
from torch.autograd import Variable
from utils.utils import or_float_tensors, xor_float_tensors, clip_tensor
import numpy as np


# helper function
def round_x(x, alpha=0.5):
    """
    rounds x by thresholding it according to alpha which can be a scalar or vector
    :param x:
    :param alpha: threshold parameter
    :return: a float tensor of 0s and 1s.
    """
    return (x > alpha).float()


def get_x0(x, is_sample=False):
    """
    Helper function to randomly initialize the the inner maximizer algos
    randomize such that the functionality is preserved.
    Functionality is preserved by maintaining the features present in x
    :param x: training sample
    :param is_sample: flag to sample randomly from feasible area or return just x
    :return: randomly sampled feasible version of x
    """
    if is_sample:
        rand_x = round_x(torch.rand(x.size()))
        if x.is_cuda:
            rand_x = rand_x.cuda()
        return or_float_tensors(x, rand_x)
    else:
        return x


def dfgsm_k(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False):
    """
    FGSM^k with deterministic rounding
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param alpha:
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to dfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
              (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def rfgsm_k(x, y, model, loss_fct, k=25, epsilon=0.02, is_report_loss_diff=False, use_sample=False):
    """
    FGSM^k with randomized rounding
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to rfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step with gradients
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    alpha = torch.rand(x_next.size())
    if x_next.is_cuda:
        alpha = alpha.cuda()
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
              (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def bga_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit gradient ascent
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to bga_k (tensor)
    """
    # some book-keeping
    sqrt_m = torch.from_numpy(np.sqrt([x.size()[1]])).float()

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
        sqrt_m = sqrt_m.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates
            x_update = (sqrt_m * (1. - 2. * x_next) * grad_data >= torch.norm(
                grad_data, 2, 1).unsqueeze(1).expand_as(x_next)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
              (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def bca_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent
    :param use_sample:
    :param is_report_loss_diff:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :return: the adversarial version of x according to bca_k (tensor)
    """
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates (can be made more efficient than this)
            aug_grad = (1. - 2. * x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
              (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def grosse_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent using gradient of output, advancing in direction of maximal change
    :param use_sample:
    :param is_report_loss_diff:
    :param loss_fct:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param k: num of steps
    :return adversarial version of x (tensor)
    """

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    output = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            grad_vars = torch.autograd.grad(output[:, 0].mean(), x_var)
            grad_data = grad_vars[0].data

            # Only consider gradients for points of 0 value
            aug_grad = (1. - x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        x_var = Variable(x_next, requires_grad=True)
        output = model(x_var)

        loss = loss_fct(output, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
              (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst

def grams(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)
    # initialize starting point
    best_x = x
    x = get_x0(x, use_sample)
    orig_x = get_x0(x, use_sample)
    
    k = 8
    # Get the amount of features
    amount_of_features = (orig_x.size()[1])
    it = 0
    while k > 0.5:
        print("Iteration " + str(it))
        print("k " + str(k))

        it += 1
        # forward pass
        x_var = Variable(x, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad = torch.autograd.grad(loss.mean(), x_var)
        
        # compute sign
        sign = torch.sign(grad[0].data)
        # compute gradient
        grad = torch.abs(grad[0].data - orig_x * grad[0].data)
        
        # returns the k largest elements of the given input tensor
        values, indices = torch.topk(grad, int(k))
        # Create binary floattensor cotaining 1, 
        # at the location of the top-k indides
        top_k_features = [0] * amount_of_features
        
        # Set the indices to 1
        indices = Variable(indices.float())
        for i in indices.data[0]:
          top_k_features[int(i)] = 1
        
        # Compute new x with k flipped bits
        top_k_features = Variable(torch.FloatTensor([top_k_features]).cuda())
        x_new = x_var + top_k_features

        # x_var = Variable(x_new, requires_grad=True)
        y_model = model(x_new)
        loss = loss_fct(y_model, y)

        x_var = Variable(best_x, requires_grad=True)
        y_model = model(x_var)
        loss_new = loss_fct(y_model, y)

        decrease_k = 1
        loss_comparision = torch.gt(loss, loss_new)
        count = 0

        for i in loss_comparision.data:
          if i == 1:
            # update best_x
            best_x[count] = x_new.data[count]
            # Boolean checks wheter the condition is met onces
            decrease_k = 0
          count += 1
        
        # No loss found which met condition
        if decrease_k == 1:
          k = k * 0.5
        else:
          x = x_new.data
          k = k * 2

    if best_x.is_cuda:
        best_x = best_x.cpu()
    
    return best_x

def inner_maximizer(x, y, model, loss_fct, iterations=100, method='natural'):
    """
    A wrapper function for the above algorithim
    :param iterations:
    :param x:
    :param y:
    :param model:
    :param loss_fct:
    :param method: one of 'dfgsm_k', 'rfgsm_k', 'bga_k', 'bca_k', 'natural
    :return: adversarial examples
    """
    if method == 'dfgsm_k':
        return dfgsm_k(x, y, model, loss_fct, k=iterations)
    elif method == 'rfgsm_k':
        return rfgsm_k(x, y, model, loss_fct, k=iterations)
    elif method == 'bga_k':
        return bga_k(x, y, model, loss_fct, k=iterations)
    elif method == 'bca_k':
        return bca_k(x, y, model, loss_fct, k=iterations)
    elif method == 'grosse':
        return grosse(x, y, model, loss_fct, k=iterations)
    elif method == 'grams':
        return grams(x, y, model, loss_fct, k=iterations)
    elif method == 'natural':
        return x
    else:
        raise Exception('No such inner maximizer algorithm')
