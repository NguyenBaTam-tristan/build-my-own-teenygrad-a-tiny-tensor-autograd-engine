"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    prod = 1
    if not shape: return 1
    for num in shape:
        prod *= num
    return prod

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    index_values = list(enumerate(values))
    sorted_values = sorted(index_values , key=lambda x: x[1])
    result = []
    for i in range(len(sorted_values)):
        result.append(sorted_values[i][0])
    return result

# Step 3 - make_op_enums
from enum import Enum, auto
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    # phép toán một ngôi: nhận 1 tensor, biến dổi từng 
    class UnaryOps(Enum):
        NEG = auto() # đổi dấu x -> -x
        RELU = auto() # hàm kích hoạt Relu: max(0, x)
        LOG = auto() # logarit tự nhiên (ln(x))
        EXP = auto() # hàm mũ cơ số e (e^x)
        SQRT = auto() # căn bậc 2
        SIGMOID = auto() # 1 / (1 + (e^(-x)))
    # phép toán 2 ngôi: nhận 2 tensor cùng or khác kích thước
    class BinaryOps(Enum):
        ADD = auto() # cộng
        SUB = auto() # trừ
        MUL = auto() # nhân
        DIV = auto() # chia
        CMPLT = auto() # so sánh return True False
        MAX = auto() # max(a, b) từng cặp
    # thu gọn chiều theo trục chỉ định
    class ReduceOps(Enum):
        SUM = auto() # tổng theo trục
        MAX = auto() # max theo trục
    # biến dổi không gian
    class MovementOps(Enum):
        RESHAPE = auto() # đỏi kích thước từ (2,3) -> (6,)
        EXPAND = auto() # kéo một kích thước độ dài 1 thành kích thước lớn hơn cho broadcast
        PERMUTE = auto() # tráo đổi thứ tự các trục
    
    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        if isinstance(np_array, np.ndarray):
            self._np = np_array
        else:
            self._np = np.array(np_array)
        
        self.shape = self._np.shape
        self.dtype = self._np.dtype

    def tolist(self):
        return self._np.tolist()

    def __repr__(self):
        return f'LazzyBuffer shape = {self.shape} dtype = {self.dtype}'

# Step 5 - lazybuffer_const (not yet solved)
# TODO: implement

# Step 6 - rand (not yet solved)
# TODO: implement

# Step 7 - lazybuffer_unary_e (not yet solved)
# TODO: implement

# Step 8 - lazybuffer_binary_e (not yet solved)
# TODO: implement

# Step 9 - lazybuffer_r (not yet solved)
# TODO: implement

# Step 10 - lazybuffer_reshape (not yet solved)
# TODO: implement

# Step 11 - lazybuffer_expand (not yet solved)
# TODO: implement

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function (not yet solved)
# TODO: implement

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

