#include <iostream>
#include <torch/torch.h>    // libtorch头文件

using namespace torch;  // libtorch命名空间
using namespace std;

int main() {
    // 分别打印CUDA、cuDNN是否可用以及可用的CUDA设备个数
    // 可以发现函数名是和PyTorch里一样的
    cout << torch::cuda::is_available() << endl;
    cout << torch::cuda::cudnn_is_available() << endl;
    cout << torch::cuda::device_count() << endl;
    return 0;
}
