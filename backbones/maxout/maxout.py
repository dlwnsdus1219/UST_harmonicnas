## Maxout 네트워크의 정의 (다양한 입력에 대해 최댓값 취하기)
from torch import nn

# 활성화 함수의 정의(입력 Tensor의 각 위치에서 여러 선형 변환의 최댓값 계산)
class Maxout(nn.Module):
    """Implements Maxout module."""
    
    def __init__(self, d, m, k):
        """Initialize Maxout object.

        Args:
            d (int): (Unused)
            m (int): Number of features remeaining after Maxout.
            k (int): Pool Size
        """
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        """Apply Maxout to inputs.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        shape = list(inputs.size())
        shape[-1] = self.d_out      # 출력 크기(m)
        shape.append(self.pool_size)        # [batch_size, m, k]
        max_dim = len(shape) - 1
        out = self.lin(inputs)      # [batch_size, m * k]
        m, _ = out.view(*shape).max(dim=max_dim)
        return m        #최종 출력은 [batch_size, m] 입니다!!

# Maxout을 여러 layer로 쌓음
class Our_Maxout(nn.Module):

    def __init__(self, first_hidden=128, number_input_feats=300, num_outputs=23, num_blocks=2, factor=2, dropout=0.3, k=4):
        super(Our_Maxout, self).__init__()
        self.op_list = [Maxout(number_input_feats, first_hidden, k), 
                        nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(dropout))]
        f_hidden = first_hidden
        for i in range(num_blocks-1):
            self.op_list.append(Maxout(f_hidden, f_hidden * factor, k))
            self.op_list.append(nn.Sequential(nn.BatchNorm1d(f_hidden * factor), nn.Dropout(dropout)))
            f_hidden = f_hidden * factor
        
        self.op_list = nn.ModuleList(self.op_list)

        # The linear layer that maps from hidden state space to output space
        self.hid2val = nn.Linear(f_hidden, num_outputs)

    def forward(self, x):
        out_list = []
        out = x
        for i in range(len(self.op_list)):      # 각 layer의 순차 적용
            op = self.op_list[i]
            out = op(out)       # 현재 연산의 적용
            if(i % 2 == 0):     # Maxout 레이어 -> 출력 저장 ㄱㄱ
                out_list.append(out)

        out = self.hid2val(out)
        out_list.append(out)

        return out_list