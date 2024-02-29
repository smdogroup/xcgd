import numpy as np
import matplotlib.pyplot as plt


def lagrange_polynomials_1d(xs):
    """
    Args:
        xs: array, nodes that defines the Lagrange bases

    Return:
        funcs: list of callables, funcs[i](x) evaluates the Lagrange basis l_i(x)
    """
    funcs = []
    for j in range(len(xs)):

        def lj(x, j=j):
            ljx = 1.0
            for m in range(len(xs)):
                if m != j:
                    ljx *= (x - xs[m]) / (xs[j] - xs[m])
            return ljx

        funcs.append(lj)
    return funcs


def lagrange_polynomials_2d(xs1, xs2):
    """
    2D Lagrange basis functions via tensor product

    Args:
        xs: array, nodes that defines the Lagrange bases

    Return:
        funcs: list of callables, funcs[i][j](x, y) evaluates l_i(x) * l_j(y),
               where l_i and l_j are Lagrange bases
    """
    li = lagrange_polynomials_1d(xs1)
    lj = lagrange_polynomials_1d(xs2)

    funcs = []
    for i in range(len(li)):
        funcs.append([])
        for j in range(len(lj)):
            funcs[i].append(lambda x, y, i=i, j=j: li[i](x) * lj[j](y))
    return funcs


def polynomials_fit_1d(p, pts):
    """
    Args:
        p: polynomial degree
        pts: list of x-coordinates
    """
    Nk = len(pts)
    Np = 1 + p
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for j in range(Np):
        Vk[:, j] = pts**j

    Ck = np.linalg.inv(Vk)
    print("[1d]condition number of Vk:", np.linalg.cond(Vk))

    funcs = []
    for i in range(Nk):

        def phi(x, i=i):
            ret = 0.0
            for j in range(Np):
                ret += Ck[j, i] * x**j
            return ret

        funcs.append(phi)
    return funcs


def polynomials_fit_2d(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """

    Nk = len(pts)
    Np_1d = 1 + p
    Np = Np_1d**2
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for i, xy in enumerate(pts):
        x = xy[0]
        y = xy[1]
        xpows = [x**j for j in range(Np_1d)]
        ypows = [y**j for j in range(Np_1d)]
        for j in range(Np_1d):
            for k in range(Np_1d):
                idx = j * Np_1d + k
                Vk[i, idx] = xpows[j] * ypows[k]

    Ck = np.linalg.inv(Vk)
    cond_Vk = np.linalg.cond(Vk)

    funcs = []
    for i in range(Nk):

        def phi(x, y, i=i):
            xpows = [x**j for j in range(Np_1d)]
            ypows = [y**j for j in range(Np_1d)]
            ret = 0.0
            for j in range(Np_1d):
                for k in range(Np_1d):
                    idx = j * Np_1d + k
                    ret += Ck[idx, i] * xpows[j] * ypows[k]
            return ret

        funcs.append(phi)

    return funcs, cond_Vk


def polynomials_fit_2d_ref_elem(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """
    pts = np.array(pts)
    pt_min = pts.min(axis=0)
    pt_max = pts.max(axis=0)
    h = pt_max - pt_min
    pts_ref = -1.0 + 2.0 * (pts - pt_min) / h

    # plt.plot(pts_ref[:, 0], pts_ref[:, 1], "o")
    # plt.grid()
    # plt.show()
    # exit()

    funcs_ref, cond_Vk = polynomials_fit_2d(p, pts_ref)

    funcs = []

    for fref in funcs_ref:

        def phi(x, y, fref=fref):
            return fref(
                -1.0 + 2.0 * (x - pt_min[0]) / h[0],
                -1.0 + 2.0 * (y - pt_min[1]) / h[1],
            )

        funcs.append(phi)

    return funcs, cond_Vk


def demo_poly_fit_1d(p=3):
    start = 40.0
    stop = p + 40.0
    num = p + 1
    xs = np.linspace(start, stop, num)
    funcs_lag = lagrange_polynomials_1d(xs)
    funcs_fit = polynomials_fit_1d(p, xs)

    x = np.linspace(start, stop, 201)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    for i, fun in enumerate(funcs_lag):
        axs[0].plot(x, fun(x), label="basis %d" % i)

    for i, fun in enumerate(funcs_fit):
        axs[1].plot(x, fun(x), label="basis %d" % i)

    for i, (fun1, fun2) in enumerate(zip(funcs_lag, funcs_fit)):
        axs[2].semilogy(x, fun1(x) - fun2(x), label="basis %d" % i)

    axs[0].set_title("Lagrange polynomials")
    axs[1].set_title("Polynomials by fit")
    axs[2].set_title("Error")

    for ax in axs:
        ax.legend()
        ax.grid()

    plt.show()
    return


def demo_poly_fit_2d(p=3, i=0, j=0):
    start = 5.0
    stop = p + 5.0
    num = p + 1

    pts = np.linspace(start, stop, num)
    funcs_lag = lagrange_polynomials_2d(pts, pts)

    pts2 = [(i, j) for i in pts for j in pts]

    funcs_fit, cond = polynomials_fit_2d(p, pts2)
    funcs_fit_ref, cond_ref = polynomials_fit_2d_ref_elem(p, pts2)

    x = np.linspace(start, stop, 201)
    y = np.linspace(start, stop, 201)
    x, y = np.meshgrid(x, y)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    axs[0].contour(x, y, funcs_lag[i][j](x, y), levels=100)
    axs[1].contour(x, y, funcs_fit[(p + 1) * i + j](x, y), levels=100)
    axs[2].contour(x, y, funcs_fit_ref[(p + 1) * i + j](x, y), levels=100)

    axs[0].set_title("Lagrange polynomials (%d, %d)" % (i, j))
    axs[1].set_title("Polynomials by fit, cond(Vk): %.2e" % cond)
    axs[2].set_title("Polynomials by fit and normalization, cond(Vk): %.2e" % cond_ref)

    plt.show()

    return


def test_gd_impl(Np_1d=2):
    pts = [2.0 * i / (Np_1d - 1.0) - 1.0 for i in range(Np_1d)]
    pts = np.array(pts)
    print(pts)
    pts2 = [(i, j) for j in pts for i in pts]
    pts2 = np.array(pts2)

    funcs_fit, cond = polynomials_fit_2d(Np_1d - 1, pts2)
    print(cond)
    exit()

    x, y = 0.39214122, -0.24213123
    for i, f in enumerate(funcs_fit):
        print("%.16f," % f(x, y))

    return


if __name__ == "__main__":
    # demo_poly_fit_1d()
    # demo_poly_fit_2d()
    test_gd_impl(6)
