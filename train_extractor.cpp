#include <torch/torch.h>
#include <iostream>
#include <numeric>
#include <vector>
#include "lyft_motion_prediction/train/train_monitor.h"
#include "lyft_motion_prediction/train/train_utils.h"
#include "lyft_motion_prediction/train/losses.h"

using namespace std;

// Function to compute forward pass for Extractor model.
// Returns:
//   preds (Tensor): reconstructed trajectory.
//   loss (Tensor): loss w.r.t. criterion.
torch::Tensor forward_extractor(
    torch::nn::Module& cvae_model,
    torch::nn::Module& extractor_model,
    torch::Tensor data,
    torch::Device device,
    function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)> criterion,
    torch::Tensor confs,
    map<string, int> cfg) {

  torch::Tensor context = data.to(device);
  torch::Tensor trajectories = sample_trajectories_batch(cvae_model, context, device, cfg).to(device);
  torch::Tensor target_availabilities = data["target_availabilities"].to(device);
  torch::Tensor targets = data["target_positions"].to(device);

  // Forward pass
  torch::Tensor preds = extractor_model.forward(trajectories);
  torch::Tensor loss = criterion(targets, preds, confs, target_availabilities);

  return loss;
}

void train_extractor(
    torch::nn::Module& cvae_model,
    torch::nn::Module& extractor_model,
    torch::data::DataLoader<torch::Tensor>& data_loader,
    torch::Tensor confs,
    torch::optim::Optimizer& optimizer,
    torch::Device device,
    map<string, int> cfg,
    bool plot_mode=true) {

  string checkpoint_path = cfg["models_checkpoint_path"];

  auto tr_it = data_loader.begin();
  vector<float> losses_train;
  vector<int> iterations;

  for (int i = 0; i < cfg["train_extractor_params"]["max_num_steps"]; i++) {
    auto data = *tr_it;

    extractor_model.train();
    torch::set_grad_enabled(true);
    // Forward
    torch::Tensor loss = forward_extractor(cvae_model, extractor_model, data, device,
                                           neg_multi_log_likelihood_batch, confs, cfg);
    // Backward
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    iterations.push_back(i);
    losses_train.push_back(loss.item().toFloat());  // mean per batch
    cout << "loss: " << loss.item().toFloat() << ", loss(avg): " << accumulate(losses_train.begin(),
