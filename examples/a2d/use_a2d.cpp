#include <a2dcore.h>

template <typename T, int spatial_dim, int dof_per_node, class BCFunc>
void test_jacobian(
    T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
    A2D::Vec<T, spatial_dim>& nrm_ref, A2D::Mat<T, spatial_dim, spatial_dim>& J,
    A2D::Vec<T, dof_per_node>& u, A2D::Mat<T, dof_per_node, spatial_dim>& grad,
    A2D::Mat<T, dof_per_node, dof_per_node>& jac_u,
    A2D::Mat<T, dof_per_node, dof_per_node * spatial_dim>& jac_mixed,
    A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>&
        jac_grad,
    const BCFunc& bc_func) {
  // Prepare quantities
  A2D::Vec<T, spatial_dim> tan_ref, nrm;
  A2D::Mat<T, spatial_dim, spatial_dim> rot;
  rot(0, 1) = -1.0;
  rot(1, 0) = 1.0;
  A2D::MatVecMult(rot, nrm_ref, tan_ref);
  A2D::MatVecMult(J, nrm_ref, nrm);

  // Create quantites
  A2D::Vec<T, dof_per_node> ub, up, uh;
  A2D::Mat<T, dof_per_node, spatial_dim> bgrad, pgrad, hgrad;
  A2D::A2DObj<T> scale_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj, ug_obj;
  A2D::A2DObj<A2D::Vec<T, dof_per_node>> ngrad_obj;
  A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
  A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

  A2D::A2DObj<A2D::Vec<T, dof_per_node>&> u_obj(u, ub, up, uh);
  A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad, bgrad,
                                                                pgrad, hgrad);
  T eta = 1.3;

  // Compute
  auto stack = A2D::MakeStack(
      A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                 JTJ_obj),
      A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
      A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
      A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
      A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
      A2D::VecDot(u_obj, u_obj, uu_obj),
      A2D::VecDot(ngrad_obj, bc_func(xloc), ngradg_obj),
      A2D::VecDot(u_obj, bc_func(xloc), ug_obj),
      A2D::Eval(weight * sqrt(scale_obj) *
                    (-ngradu_obj + ngradg_obj + eta * (0.5 * uu_obj - ug_obj)),
                output_obj));
  output_obj.bvalue() = 1.0;

  // Note: Extract Hessian w.r.t. ∇u: ∂2e/∂(∇_x)uq2 is zero
  // stack.hextract(pgrad, hgrad, jac_grad);
  //
  // Extract the Hessian w.r.t. u
  stack.hextract(up, uh, jac_u);

  for (int i = 0; i < dof_per_node; i++) {
    up.zero();
    hgrad.zero();
    stack.hzero();

    up[i] = 1.0;

    stack.hforward();
    stack.hreverse();

    // Extract the mixed Hessian w.r.t. u and ∇u:
    for (int j = 0; j < dof_per_node; j++) {
      for (int k = 0; k < spatial_dim; k++) {
        jac_mixed(i, j * spatial_dim + k) = hgrad(j, k);
      }
    }
  }
}

int main() {
  using T = double;
  int constexpr spatial_dim = 2;
  int constexpr dof_per_node = 2;

  auto bc_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    return A2D::Vec<T, spatial_dim>{};
  };

  T weight = 1.2;
  T x = 3.4;
  A2D::Vec<T, spatial_dim> xloc;
  A2D::Vec<T, spatial_dim> nrm_ref;
  A2D::Mat<T, spatial_dim, spatial_dim> J;
  A2D::Vec<T, dof_per_node> u;
  A2D::Mat<T, dof_per_node, spatial_dim> grad;
  A2D::Mat<T, dof_per_node, dof_per_node> jac_u;
  A2D::Mat<T, dof_per_node, dof_per_node * spatial_dim> jac_mixed;
  A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim> jac_grad;

  test_jacobian<T, spatial_dim, dof_per_node, typeof(bc_fun)>(
      weight, x, xloc, nrm_ref, J, u, grad, jac_u, jac_mixed, jac_grad, bc_fun);
}
