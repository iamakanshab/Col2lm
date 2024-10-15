#include <vector>
#include <cstring>
#include <algorithm>

class Col2Im5D {
public:
    static void compute(
        const float* columns,
        float* output,
        const int* input_shape,
        const int* block_shape,
        const int* dilations,
        const int* pads,
        const int* strides
    ) {
        int N = input_shape[0], C = input_shape[1], D = input_shape[2], H = input_shape[3], W = input_shape[4];
        int kD = block_shape[0], kH = block_shape[1], kW = block_shape[2];

        // Calculate output dimensions
        int out_depth = (D + pads[0] + pads[3] - (dilations[0] * (kD - 1) + 1)) / strides[0] + 1;
        int out_height = (H + pads[1] + pads[4] - (dilations[1] * (kH - 1) + 1)) / strides[1] + 1;
        int out_width = (W + pads[2] + pads[5] - (dilations[2] * (kW - 1) + 1)) / strides[2] + 1;

        // Initialize output array with padding
        int padded_D = D + pads[0] + pads[3];
        int padded_H = H + pads[1] + pads[4];
        int padded_W = W + pads[2] + pads[5];
        std::vector<float> padded_output(N * C * padded_D * padded_H * padded_W, 0);

        // Perform the col2im operation
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int kd = 0; kd < kD; ++kd) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int d_offset = kd * dilations[0];
                            int h_offset = kh * dilations[1];
                            int w_offset = kw * dilations[2];

                            for (int od = 0; od < out_depth; ++od) {
                                for (int oh = 0; oh < out_height; ++oh) {
                                    for (int ow = 0; ow < out_width; ++ow) {
                                        int d_in = od * strides[0] + d_offset;
                                        int h_in = oh * strides[1] + h_offset;
                                        int w_in = ow * strides[2] + w_offset;

                                        if (d_in < padded_D && h_in < padded_H && w_in < padded_W) {
                                            int col_idx = ((((n * C + c) * kD + kd) * kH + kh) * kW + kw) * out_depth * out_height * out_width
                                                          + (od * out_height + oh) * out_width + ow;
                                            int out_idx = ((n * C + c) * padded_D + d_in) * padded_H * padded_W
                                                          + h_in * padded_W + w_in;
                                            padded_output[out_idx] += columns[col_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remove padding
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int d = 0; d < D; ++d) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int out_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                            int padded_idx = ((n * C + c) * padded_D + (d + pads[0])) * padded_H * padded_W
                                             + (h + pads[1]) * padded_W + (w + pads[2]);
                            output[out_idx] = padded_output[padded_idx];
                        }
                    }
                }
            }
        }
    }
};

// Example usage
int main() {
    // Define input parameters
    int input_shape[] = {2, 3, 10, 10, 10}; // N, C, D, H, W
    int block_shape[] = {3, 3, 3}; // kD, kH, kW
    int dilations[] = {1, 1, 1};
    int pads[] = {1, 1, 1, 1, 1, 1};
    int strides[] = {1, 1, 1};

    // Calculate the shape of the columns tensor
    int N = input_shape[0], C = input_shape[1], D = input_shape[2], H = input_shape[3], W = input_shape[4];
    int kD = block_shape[0], kH = block_shape[1], kW = block_shape[2];
    int out_depth = (D + pads[0] + pads[3] - (dilations[0] * (kD - 1) + 1)) / strides[0] + 1;
    int out_height = (H + pads[1] + pads[4] - (dilations[1] * (kH - 1) + 1)) / strides[1] + 1;
    int out_width = (W + pads[2] + pads[5] - (dilations[2] * (kW - 1) + 1)) / strides[2] + 1;
    int columns_size = N * C * kD * kH * kW * out_depth * out_height * out_width;

    // Create input columns (you would normally get this from your network)
    std::vector<float> columns(columns_size);
    // Fill columns with some data (e.g., random values)

    // Create output tensor
    std::vector<float> output(N * C * D * H * W);

    // Perform col2im operation
    Col2Im5D::compute(columns.data(), output.data(), input_shape, block_shape, dilations, pads, strides);

    // Use the output...

    return 0;
}
