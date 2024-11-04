#include <a2dcore.h>

#include <complex>

#include "ad/a2dgreenstrain.h"
#include "ad/a2disotropic.h"

#define PI 3.14159265358979323846

template <typename T, int dof_per_node, int spatial_dim>
T stress_component(T E, T nu, A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                   int I, int J) {
  static_assert(dof_per_node == spatial_dim);
  T mu = 0.5 * E / (1.0 + nu);
  T lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  A2D::SymMat<T, spatial_dim> strain, stress;
  A2D::Vec<T, spatial_dim> t;

  A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, strain);
  A2D::SymIsotropic(mu, lambda, strain, stress);

  return stress(I, J);
}

template <int dof_per_node, int spatial_dim>
A2D::Mat<double, dof_per_node, spatial_dim> stress_component_grad(
    double E, double nu, A2D::Mat<double, dof_per_node, spatial_dim>& grad,
    int I, int J) {
  double mu = 0.5 * E / (1.0 + nu);
  double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  A2D::ADObj<double> output;
  A2D::ADObj<A2D::Vec<double, spatial_dim>> t_obj;
  A2D::ADObj<A2D::Mat<double, dof_per_node, spatial_dim>> grad_obj(grad);
  A2D::ADObj<A2D::SymMat<double, spatial_dim>> strain_obj, stress_obj;

  auto stack = A2D::MakeStack(
      A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, strain_obj),
      A2D::SymIsotropic(mu, lambda, strain_obj, stress_obj));

  stress_obj.bvalue()(I, J) = 1.0;
  stack.reverse();

  return grad_obj.bvalue();
}

template <int dof_per_node, int spatial_dim>
A2D::Mat<double, dof_per_node, spatial_dim> stress_component_grad_cs(
    double E, double nu, A2D::Mat<double, dof_per_node, spatial_dim>& grad,
    int I, int J) {
  using T = std::complex<double>;

  double h = 1e-30;
  A2D::Mat<T, dof_per_node, spatial_dim> gradc;
  A2D::Vec<T, spatial_dim> phic, psic;

  for (int j = 0; j < spatial_dim; j++) {
    for (int i = 0; i < dof_per_node; i++) {
      gradc(i, j) = T(grad(i, j));
    }
  }

  A2D::Mat<double, dof_per_node, spatial_dim> outgrad;
  for (int i = 0; i < dof_per_node; i++) {
    for (int j = 0; j < spatial_dim; j++) {
      gradc(i, j) += T(0.0, h);
      T stress =
          stress_component<T, dof_per_node, spatial_dim>(E, nu, gradc, I, J);
      outgrad(i, j) = stress.imag() / h;
      gradc(i, j) -= T(0.0, h);
    }
  }
  return outgrad;
}

template <typename T, int dof_per_node, int spatial_dim>
A2D::SymMat<T, spatial_dim> compute_stress(T E, T nu,
                                           A2D::Vec<T, spatial_dim>& xloc) {
  A2D::Vec<T, spatial_dim> u;

  T k = 1.9 * PI;
  T mu = 0.5 * E / (1.0 + nu);
  T lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  u(0) = sin(k * xloc(0)) * sin(k * xloc(1));
  u(1) = cos(k * xloc(0)) * cos(k * xloc(1));

  A2D::Mat<T, dof_per_node, spatial_dim> grad;
  T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
  T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
  T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
  T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

  grad(0, 0) = ux;
  grad(0, 1) = uy;
  grad(1, 0) = vx;
  grad(1, 1) = vy;

  A2D::SymMat<T, spatial_dim> strain, stress;
  A2D::Vec<T, spatial_dim> t;

  A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, strain);
  A2D::SymIsotropic(mu, lambda, strain, stress);

  return stress;
}

template <typename T, int spatial_dim>
A2D::Vec<T, spatial_dim> compute_intf_cs(T E, T nu,
                                         A2D::Vec<T, spatial_dim>& xloc) {
  constexpr int dof_per_node = spatial_dim;

  using T2 = std::complex<T>;
  double h = 1e-30;

  A2D::Vec<T2, spatial_dim> xlocc;
  for (int i = 0; i < spatial_dim; i++) {
    xlocc(i) = xloc(i);
  }

  A2D::Vec<double, spatial_dim> intf;
  for (int i = 0; i < spatial_dim; i++) {
    for (int j = 0; j < spatial_dim; j++) {
      xlocc(j) += T2(0.0, h);
      A2D::SymMat<T2, spatial_dim> stress =
          compute_stress<T2, dof_per_node, spatial_dim>(E, nu, xlocc);
      xlocc(j) -= T2(0.0, h);
      intf(i) -= stress(i, j).imag() / h;
    }
  }
  return intf;
}

int test_stress_ad(int argc, char* argv[]) {
  double E = 100.0, nu = 0.3;
  constexpr int dof_per_node = 2;
  constexpr int spatial_dim = 2;

  A2D::Mat<double, dof_per_node, spatial_dim> grad, outgrad_exact, outgrad_cs;
  grad(0, 0) = 2.5;
  grad(0, 1) = 3.6;
  grad(1, 0) = 0.1;
  grad(1, 1) = -0.2;

  int I = std::atoi(argv[1]), J = std::atoi(argv[2]);
  outgrad_exact = stress_component_grad(E, nu, grad, I, J);
  outgrad_cs = stress_component_grad_cs(E, nu, grad, I, J);

  for (int i = 0; i < dof_per_node; i++) {
    for (int j = 0; j < dof_per_node; j++) {
      std::printf("stress(%d, %d): exact: %20.10f, cs: %20.10f\n", i, j,
                  outgrad_exact(i, j), outgrad_cs(i, j));
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  // test_stress_ad(argc, argv);

  double E = 100.0, nu = 0.3;

  constexpr int dof_per_node = 2;
  constexpr int spatial_dim = 2;
  using T = double;

  auto elasticity_int_fun = [E, nu](const A2D::Vec<T, spatial_dim>& xloc) {
    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    double k = 1.9 * PI;
    double k2 = k * k;

    A2D::Mat<T, dof_per_node, spatial_dim> grad;
    T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
    T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

    grad(0, 0) = ux;
    grad(0, 1) = uy;
    grad(1, 0) = vx;
    grad(1, 1) = vy;

    T uxx = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T uxy = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyx = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyy = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vxx = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T vxy = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyx = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyy = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));

    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>> grad_obj(grad);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj));

    // Spartials(i, j) = ∂S(i, j)/∂x(j)
    A2D::Mat<T, dof_per_node, spatial_dim> Spartials;

    for (int i = 0; i < spatial_dim; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        grad_obj.bvalue().zero();
        E_obj.bvalue().zero();
        S_obj.bvalue().zero();
        S_obj.bvalue()(i, j) = 1.0;

        stack.reverse();

        // ∂S(i, j)/∂x(j) = ∂S(i, j)/∂grad * ∂grad/∂x(j)
        auto& bgrad = grad_obj.bvalue();

        if (j == 0) {
          Spartials(i, j) = bgrad(0, 0) * uxx + bgrad(0, 1) * uyx +
                            bgrad(1, 0) * vxx + bgrad(1, 1) * vyx;
        } else {
          Spartials(i, j) = bgrad(0, 0) * uxy + bgrad(0, 1) * uyy +
                            bgrad(1, 0) * vxy + bgrad(1, 1) * vyy;
        }
      }
    }

    A2D::Vec<T, dof_per_node> intf;
    intf(0) = -(Spartials(0, 0) + Spartials(0, 1));
    intf(1) = -(Spartials(1, 0) + Spartials(1, 1));
    return intf;
  };

  A2D::Vec<double, spatial_dim> xloc;
  xloc(0) = 0.14;
  xloc(1) = 0.32;
  auto intf = elasticity_int_fun(xloc);
  auto intf_cs = compute_intf_cs<double>(E, nu, xloc);

  for (int i = 0; i < spatial_dim; i++) {
    std::printf("[i=%d]σij,j: %20.10e, fi: %20.10e, σij,j + fi: %20.10e\n", i,
                -intf_cs(i), intf(i), intf(i) - intf_cs(i));
  }
}
