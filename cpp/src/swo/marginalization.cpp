#include "msckf/swo/marginalization.h"

#include <cmath>
#include <map>

#include <Eigen/Eigenvalues>

#include "msckf/swo/quaternion_manifold.h"

namespace msckf::swo {

namespace {

constexpr double kEigenvalueEpsilon = 1e-8;

// d Minus(q, q_lin) / d q, with Minus = Log(q_lin^-1 q) ~ 2 vec(q_lin^-1 q).
Eigen::Matrix<double, 3, 4> minus_jacobian(const Eigen::Quaterniond& q_lin) {
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J;
  RightQuaternionManifold().MinusJacobian(q_lin.coeffs().data(), J.data());
  return J;
}

}  // namespace

MarginalizationFactor::MarginalizationFactor(std::vector<Block> blocks, Eigen::MatrixXd J_p,
                                             Eigen::VectorXd e)
    : blocks_(std::move(blocks)), J_p_(std::move(J_p)), e_(std::move(e)) {
  int offset = 0;
  for (const Block& b : blocks_) {
    mutable_parameter_block_sizes()->push_back(b.ambient_size);
    tangent_offset_.push_back(offset);
    offset += b.is_quaternion ? 3 : b.ambient_size;
  }
  set_num_residuals(static_cast<int>(e_.size()));
}

std::vector<double*> MarginalizationFactor::block_pointers() const {
  std::vector<double*> out;
  for (const Block& b : blocks_) {
    out.push_back(b.ptr);
  }
  return out;
}

bool MarginalizationFactor::Evaluate(double const* const* parameters, double* residuals,
                                     double** jacobians) const {
  // dx = x [-] x_lin, block by block, in tangent coordinates.
  Eigen::VectorXd dx(J_p_.cols());
  for (size_t k = 0; k < blocks_.size(); ++k) {
    const Block& b = blocks_[k];
    if (b.is_quaternion) {
      const Eigen::Map<const Eigen::Quaterniond> q(parameters[k]);
      const Eigen::Map<const Eigen::Quaterniond> q_lin(b.x_lin.data());
      dx.segment<3>(tangent_offset_[k]) = quaternion_log(Eigen::Quaterniond(q_lin.inverse() * q));
    } else {
      dx.segment(tangent_offset_[k], b.ambient_size) =
          Eigen::Map<const Eigen::VectorXd>(parameters[k], b.ambient_size) - b.x_lin;
    }
  }
  Eigen::Map<Eigen::VectorXd>(residuals, e_.size()) = e_ + J_p_ * dx;

  if (jacobians != nullptr) {
    for (size_t k = 0; k < blocks_.size(); ++k) {
      if (jacobians[k] == nullptr) {
        continue;
      }
      const Block& b = blocks_[k];
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(
          jacobians[k], e_.size(), b.ambient_size);
      if (b.is_quaternion) {
        const Eigen::Map<const Eigen::Quaterniond> q_lin(b.x_lin.data());
        J = J_p_.middleCols<3>(tangent_offset_[k]) * minus_jacobian(q_lin);
      } else {
        J = J_p_.middleCols(tangent_offset_[k], b.ambient_size);
      }
    }
  }
  return true;
}

std::unique_ptr<MarginalizationFactor> Marginalizer::marginalize(
    const std::set<double*>& drop, const std::set<double*>& quaternion_blocks) const {
  struct BlockInfo {
    int ambient = 0;
    int tangent = 0;
    int offset = 0;   // in the stacked tangent vector
  };
  // Removed blocks first, then kept ones, each group in the order the blocks
  // first appear in the added factors. (Ordering by pointer would change from
  // run to run, and so would the rounding.)
  std::map<double*, BlockInfo> infos;
  std::vector<double*> dropped_order, kept_order;
  for (const ResidualBlockInfo& f : factors_) {
    const std::vector<int>& sizes = f.cost->parameter_block_sizes();
    for (size_t k = 0; k < f.blocks.size(); ++k) {
      double* ptr = f.blocks[k];
      if (infos.count(ptr)) {
        continue;
      }
      BlockInfo info;
      info.ambient = sizes[k];
      info.tangent = quaternion_blocks.count(ptr) ? 3 : sizes[k];
      infos.emplace(ptr, info);
      (drop.count(ptr) ? dropped_order : kept_order).push_back(ptr);
    }
  }
  int m = 0;
  for (double* ptr : dropped_order) {
    infos[ptr].offset = m;
    m += infos[ptr].tangent;
  }
  int n = m;
  for (double* ptr : kept_order) {
    infos[ptr].offset = n;
    n += infos[ptr].tangent;
  }
  const int k_size = n - m;
  if (k_size == 0) {
    return nullptr;
  }

  // Linearize every factor at the current estimate: H = sum J^T J, g = sum J^T r.
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
  const RightQuaternionManifold manifold;
  for (const ResidualBlockInfo& f : factors_) {
    const int nr = f.cost->num_residuals();
    const std::vector<int>& sizes = f.cost->parameter_block_sizes();
    Eigen::VectorXd r(nr);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J_amb;
    std::vector<double*> J_ptrs;
    for (int size : sizes) {
      J_amb.emplace_back(nr, size);
    }
    for (auto& J : J_amb) {
      J_ptrs.push_back(J.data());
    }
    f.cost->Evaluate(f.blocks.data(), r.data(), J_ptrs.data());

    // Tangent-space Jacobians: J_tangent = J_ambient * PlusJacobian.
    std::vector<Eigen::MatrixXd> J_tan(f.blocks.size());
    for (size_t k = 0; k < f.blocks.size(); ++k) {
      if (quaternion_blocks.count(f.blocks[k])) {
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> P;
        manifold.PlusJacobian(f.blocks[k], P.data());
        J_tan[k] = J_amb[k] * P;
      } else {
        J_tan[k] = J_amb[k];
      }
    }

    // Robust loss: rescale residual and Jacobian like ceres' Corrector does,
    // so the prior matches what the optimizer actually minimized.
    if (f.loss != nullptr) {
      const double sq_norm = r.squaredNorm();
      double rho[3];
      f.loss->Evaluate(sq_norm, rho);
      const double sqrt_rho1 = std::sqrt(rho[1]);
      double residual_scaling = sqrt_rho1;
      double alpha_sq_norm = 0.0;
      if (sq_norm > 0.0 && rho[2] > 0.0) {
        const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        const double alpha = 1.0 - std::sqrt(D);
        residual_scaling = sqrt_rho1 / (1.0 - alpha);
        alpha_sq_norm = alpha / sq_norm;
      }
      for (Eigen::MatrixXd& J : J_tan) {
        J = sqrt_rho1 * (J - alpha_sq_norm * r * (r.transpose() * J));
      }
      r *= residual_scaling;
    }

    auto info_of = [&](double* ptr) -> const BlockInfo& { return infos.at(ptr); };
    for (size_t a = 0; a < f.blocks.size(); ++a) {
      const BlockInfo& ia = info_of(f.blocks[a]);
      g.segment(ia.offset, ia.tangent) += J_tan[a].transpose() * r;
      for (size_t b = 0; b < f.blocks.size(); ++b) {
        const BlockInfo& ib = info_of(f.blocks[b]);
        H.block(ia.offset, ib.offset, ia.tangent, ib.tangent) +=
            J_tan[a].transpose() * J_tan[b];
      }
    }
  }

  // Schur complement. H_mm is inverted through its eigen-decomposition so a
  // rank-deficient block (e.g. a weakly observed depth) does not blow up.
  Eigen::MatrixXd H_star;
  Eigen::VectorXd g_star;
  if (m > 0) {
    const Eigen::MatrixXd H_mm = 0.5 * (H.topLeftCorner(m, m) + H.topLeftCorner(m, m).transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_mm);
    const Eigen::VectorXd inv_vals =
        (es.eigenvalues().array() > kEigenvalueEpsilon)
            .select(es.eigenvalues().array().inverse(), 0.0);
    const Eigen::MatrixXd H_mm_inv =
        es.eigenvectors() * inv_vals.asDiagonal() * es.eigenvectors().transpose();
    const Eigen::MatrixXd H_km = H.block(m, 0, k_size, m);
    H_star = H.bottomRightCorner(k_size, k_size) - H_km * H_mm_inv * H_km.transpose();
    g_star = g.tail(k_size) - H_km * H_mm_inv * g.head(m);
  } else {
    H_star = H.bottomRightCorner(k_size, k_size);
    g_star = g.tail(k_size);
  }

  // H* = U S U^T  ->  J_p = S^1/2 U^T, e = S^-1/2 U^T g* (positive part only).
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (H_star + H_star.transpose()));
  std::vector<int> keep_idx;
  for (int i = 0; i < es.eigenvalues().size(); ++i) {
    if (es.eigenvalues()(i) > kEigenvalueEpsilon) {
      keep_idx.push_back(i);
    }
  }
  Eigen::MatrixXd J_p(keep_idx.size(), k_size);
  Eigen::VectorXd e(keep_idx.size());
  for (size_t row = 0; row < keep_idx.size(); ++row) {
    const double s = es.eigenvalues()(keep_idx[row]);
    const Eigen::VectorXd u = es.eigenvectors().col(keep_idx[row]);
    J_p.row(row) = std::sqrt(s) * u.transpose();
    e(row) = u.dot(g_star) / std::sqrt(s);
  }

  std::vector<MarginalizationFactor::Block> blocks;
  for (double* ptr : kept_order) {
    MarginalizationFactor::Block b;
    b.ptr = ptr;
    b.ambient_size = infos[ptr].ambient;
    b.is_quaternion = quaternion_blocks.count(ptr) > 0;
    b.x_lin = Eigen::Map<const Eigen::VectorXd>(ptr, b.ambient_size);
    blocks.push_back(b);
  }
  return std::make_unique<MarginalizationFactor>(std::move(blocks), std::move(J_p), std::move(e));
}

}  // namespace msckf::swo
