import torch
import warnings

warnings.filterwarnings("ignore")

# Print the number of FLOPs
def print_model_parm_flops(model, input, detail=False):
    list_conv = []
    list_linear = []
    list_bn = []
    list_relu = []
    list_pooling = []

    def conv_hook(self, input, output):
        kernel_ops = (self.in_channels / self.groups) * 2 - 1  # add operations is one less to the mul operations
        for i in self.kernel_size:
            kernel_ops *= i
        bias_ops = 1 if self.bias is not None else 0

        params = kernel_ops + bias_ops
        flops = params * output[0].nelement()

        list_conv.append(flops)

    def linear_hook(self, input, output):
        weight_ops = (2 * self.in_features - 1) * output.nelement()
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        list_linear.append(flops)

    def bn_hook(self, input, output):
        # (x-x')/σ one sub op and one div op
        # and the shift γ and β
        list_bn.append(input[0].nelement() / input[0].size(0) * 4)

    def relu_hook(self, input, output):
        # every input's element need to cmp with 0
        list_relu.append(input[0].nelement() / input[0].size(0))

    def max_pooling_hook(self, input, output):
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D max pooling
                kernel_ops *= self.kernel_size
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def avg_pooling_hook(self, input, output):
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D pooling
                kernel_ops *= self.kernel_size
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adaavg_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adamax_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.MaxPool3d):
                net.register_forward_hook(max_pooling_hook)
            if isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(avg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveAvgPool2d) or isinstance(net, torch.nn.AdaptiveAvgPool3d):
                net.register_forward_hook(adaavg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveMaxPool2d) or isinstance(net, torch.nn.AdaptiveMaxPool3d):
                net.register_forward_hook(adamax_pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)
    total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
    print('Number of FLOPs:', float(total_flops / 1e9))

    if detail:
        print('Conv FLOPs:', float(sum(list_conv) / 1e9))
        print('Linear FLOPs:', float(sum(list_linear) / 1e9))
        print('Batch Norm FLOPs:', float(sum(list_bn) / 1e9))
        print('ReLU FLOPs:', float(sum(list_relu) / 1e9))
        print('Pooling FLOPs:', float(sum(list_pooling) / 1e9))

    return float(total_flops / 1e9)
