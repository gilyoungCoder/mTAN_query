# define model
import torch
import numbers
import warnings
import math

class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=128, \
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0, device = 'cuda'):
        super(GRUD, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size)).to(self.device)
        self.hzeros = torch.autograd.Variable(torch.zeros(hidden_size)).to(self.device)

        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional

        
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        
        
        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        # w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # w_mz = torch.nn.Parameter(torch.Tensor(input_size))

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # w_mr = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # w_mh = torch.nn.Parameter(torch.Tensor(input_size))

        # y (output)
        # w_hy = torch.nn.Parameter(torch.Tensor(output_size, hidden_size))

        # bias
        # b_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        # b_y = torch.nn.Parameter(torch.Tensor(output_size))

        layer_params = (w_dg_h,\
                        w_xz, w_hz, \
                        w_xr, w_hr, \
                        w_xh, w_hh, \
                       \
                        b_dg_h, b_z, b_r, b_h)

        param_names = ['weight_dg_h',\
                       'weight_xz', 'weight_hz',\
                       'weight_xr', 'weight_hr',\
                       'weight_xh', 'weight_hh',]
        if bias:
            param_names += ['bias_dg_h',\
                            'bias_z',\
                            'bias_r',\
                            'bias_h'] 
        
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)
        
        self.to(self.device)
        self.reset_parameters()
        


    def _apply(self, fn):
        ret = super()._apply(fn)
        # self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
        # for name in self._all_weights[0]:
        #     param = getattr(self, name)
        #     torch.nn.init.uniform_(param, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
    
    
    def __setstate__(self, d):
        super(GRUD, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h',\
                   'weight_xz', 'weight_hz','weight_mz',\
                   'weight_xr', 'weight_hr','weight_mr',\
                   'weight_xh', 'weight_hh','weight_mh',\
                   'weight_hy',\
                   'bias_dg_x', 'bias_dg_h',\
                   'bias_z', 'bias_r', 'bias_h','bias_y']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
    
    def forward(self, value, delta):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        # X = torch.squeeze(input[0]).to(self.device) # .size = (33,49)
        # Mask = torch.squeeze(input[1]).to(self.device) # .size = (33,49)
        # Delta = torch.squeeze(input[2]).to(self.device) # .size = (33,49)
        # Hidden_State = torch.autograd.Variable(torch.zeros(self.input_size)).to(self.device)
        
        # step_size = X.size(1) # 49
        # #print('step size : ', step_size)
        # 50 x 203 x 41
        batch_size, _, num_steps = value.size()
        # 각 데이터 유형 분리
        X = value
        # Mask = mask
        Delta = delta
        # X = input[:, 0, :, :]  # size: (batch_size, input_size, num_steps)
        # Mask = input[:, 1, :, :]  # size: (batch_size, input_size, num_steps)
        # Delta = input[:, 2, :, :]  # size: (batch_size, input_size, num_steps)
        # Last_x = input[:, 3, :, :]
        # 초기 hidden state 설정
        Hidden_State = torch.autograd.Variable(torch.zeros(batch_size, self.hidden_size)).to(self.device)
        
        output = None
        h = Hidden_State

        # decay rates gamma
        # w_dg_x = getattr(self, 'weight_dg_x')
        w_dg_h = getattr(self, 'weight_dg_h')

        #z
        w_xz = getattr(self, 'weight_xz')
        w_hz = getattr(self, 'weight_hz')
        # w_mz = getattr(self, 'weight_mz')

        # r
        w_xr = getattr(self, 'weight_xr')
        w_hr = getattr(self, 'weight_hr')
        # w_mr = getattr(self, 'weight_mr')

        # h_tilde
        w_xh = getattr(self, 'weight_xh')
        w_hh = getattr(self, 'weight_hh')
        # w_mh = getattr(self, 'weight_mh')

        # bias
        # b_dg_x = getattr(self, 'bias_dg_x')
        b_dg_h = getattr(self, 'bias_dg_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')
        
        output = torch.zeros(batch_size, num_steps, self.hidden_size).to(self.device)

        for step in range(num_steps):
            
            # x = torch.squeeze(X[:,layer:layer+1])
            # m = torch.squeeze(Mask[:,layer:layer+1])
            # d = torch.squeeze(Delta[:,layer:layer+1])
            
            x = X[:, :, step]  # size: (batch_size, input_size)
            # m = Mask[:, :, step]  # size: (batch_size, input_size)
            d = Delta[step]  # size: (batch_size, input_size)
            # last_observed_x = Last_x[:, :, step]
            # d = torch.tile(dval, x.size()).to(self.device)

            # gamma_x = torch.exp(-torch.max(self.zeros.to(self.device), (w_dg_x * d + b_dg_x)))
            gamma_h = torch.exp(-torch.max(self.hzeros, (w_dg_h * d + b_dg_h)))



            # #(4)
            # gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
            # gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

            #(5)
            # x = m * x + (1 - m) * (gamma_x * last_observed_x+ (1 - gamma_x) * self.x_mean)
            # x = m * x + (1 - m) * (gamma_x * x+ (1 - gamma_x) * self.x_mean)


            #(6)
            if self.dropout == 0:
                h = gamma_h * h

                z = torch.sigmoid((torch.matmul(x, w_xz) + torch.matmul(h, w_hz) + b_z))
                r = torch.sigmoid((torch.matmul(x, w_xr) + torch.matmul(h, w_hr) + b_r))
                h_tilde = torch.tanh((torch.matmul(x, w_xh) + torch.matmul(r*h, w_hh) + b_h))

                h = (1 - z) * h + z * h_tilde

            # elif self.dropout_type == 'Moon':
            #     '''
            #     RNNDROP: a novel dropout for rnn in asr(2015)
            #     '''
            #     h = gamma_h * h

            #     z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
            #     r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))

            #     h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

            #     h = (1 - z) * h + z * h_tilde
            #     dropout = torch.nn.Dropout(p=self.dropout)
            #     h = dropout(h)

            # elif self.dropout_type == 'Gal':
            #     '''
            #     A Theoretically grounded application of dropout in recurrent neural networks(2015)
            #     '''
            #     dropout = torch.nn.Dropout(p=self.dropout)
            #     h = dropout(h)

            #     h = gamma_h * h

            #     z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
            #     r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
            #     h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                # h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h * h

                z = torch.sigmoid((torch.matmul(x, w_xz) + torch.matmul(h, w_hz) + b_z))
                r = torch.sigmoid((torch.matmul(x, w_xr) + torch.matmul(h, w_hr) + b_r))
                h_tilde = torch.tanh((torch.matmul(x, w_xh) + torch.matmul(r*h, w_hh) + b_h))
                

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z)* h + z*h_tilde

            else:
                print("error case!")
                # h = gamma_h * h

                # z = torch.sigmoid((w_xz*x + w_hz*h + b_z))
                # r = torch.sigmoid((w_xr*x + w_hr*h + b_r))
                # h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + b_h))

                # h = (1 - z) * h + z * h_tilde
            
            output[:, step, :] = h

            
        # w_hy = getattr(self, 'weight_hy')
        # w_hy = w_hy.to(torch.float32)
        # b_y = getattr(self, 'bias_y')
        # b_y = b_y.to(torch.float32)
        h = h
        
        return output, h