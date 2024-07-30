#include<string>

struct Config {
public:
	static const int use_amp = 1;
	static const int pad_id = 3;
	static const int d_model = 512;
	static const int d_ff = 2048;
	static const int num_layer = 16;
	static const int num_head = 16;
	static const int head_dim = d_model / num_head;
	static const int batch_size = 32;
	static const int accum_iter = 2;
	static const int adam_lr = 3e-4;
	static constexpr  float adam_beta1 = 0.9;
	static constexpr  float adam_beta2 = 0.95;
	static constexpr  float adam_weight_decay = 0.1;
	static const int warmup_steps = 2000;
	static constexpr  float lr_min = 3e-5;
	static constexpr  float dropout = 0.1;
	static constexpr  float label_smoothing = 0.1;
	static constexpr  float grad_clip = 1.0;
	inline static const std::string model_path = "model.pt";
	static const int sram_size = 128 * 1024;
	static const int Bc = 32;
	static const int Br = 32;
};
