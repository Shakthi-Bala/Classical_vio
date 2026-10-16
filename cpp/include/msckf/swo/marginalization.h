#pragma once

#include <memory>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>

// Marginalization for the sliding window: turn every factor that touches the
// states being removed into one linear (Gaussian) prior on the states that stay.
//
// With all residuals stacked and linearized at the current estimate,
//   cost(dx) = |r + J dx|^2,   H = J^T J,   g = J^T r,
// split into removed (m) and kept (k) states. Minimizing over dx_m gives the
// Schur complement
//   H* = H_kk - H_km H_mm^-1 H_mk,   g* = g_k - H_km H_mm^-1 g_m.
// The prior factor reproduces that quadratic as a residual:
//   r_prior(x) = e + J_p (x [-] x_lin),  J_p^T J_p = H*,  J_p^T e = g*.

namespace msckf::swo {

// A residual block as it was added to the problem: cost, optional robust loss,
// and its parameter blocks (raw pointers into the estimator's state).
struct ResidualBlockInfo {
  ceres::CostFunction* cost = nullptr;   // not owned
  ceres::LossFunction* loss = nullptr;   // not owned, may be null
  std::vector<double*> blocks;
};

// The linear prior left behind by a marginalization step.
class MarginalizationFactor final : public ceres::CostFunction {
 public:
  struct Block {
    double* ptr;                // the estimator's parameter block
    int ambient_size;           // 4 for quaternions, else the block size
    bool is_quaternion;
    Eigen::VectorXd x_lin;      // linearization point (ambient)
  };

  MarginalizationFactor(std::vector<Block> blocks, Eigen::MatrixXd J_p, Eigen::VectorXd e);

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

  const std::vector<Block>& blocks() const { return blocks_; }
  std::vector<double*> block_pointers() const;

 private:
  std::vector<Block> blocks_;
  std::vector<int> tangent_offset_;   // column offset of each block in J_p
  Eigen::MatrixXd J_p_;               // rank x (sum of tangent sizes)
  Eigen::VectorXd e_;
};

class Marginalizer {
 public:
  void add(const ResidualBlockInfo& info) { factors_.push_back(info); }

  // Removes `drop` (parameter-block pointers) and returns the prior on every
  // other block the added factors touch. `quaternion_blocks` lists the blocks
  // that use the RightQuaternionManifold. Returns null if nothing is kept.
  std::unique_ptr<MarginalizationFactor> marginalize(
      const std::set<double*>& drop, const std::set<double*>& quaternion_blocks) const;

 private:
  std::vector<ResidualBlockInfo> factors_;
};

}  // namespace msckf::swo
