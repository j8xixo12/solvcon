#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 *
 * This file implements the support for gradient element that is used to
 * calculate the derivative.  Solver<NDIM>::calc_dsoln is implemented by using
 * the gradient element.
 */

#include <cstdint>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"
#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

/**
 * Metadata of gradient elements.
 */
struct GradientMeta {

    /**
     * Maximum number of fundamental gradient element (GE) that a cell can
     * have.
     */
    static constexpr size_t MFGE = 8;

    index_type cltype;
    index_type clnfc;
    index_type nsub; // the number of sub-GCEs is greater than or equal to the number of faces in a cell
    real_type nsub_inverse;
    index_type (*pface)[3]; 

}; /* end struct GradientMeta */

namespace detail {

/**
 * Singleton to initialize GradientMeta.  Non-copyable and non-moveable.
 */
class GradientMetaGroup {

public:

    GradientMetaGroup(GradientMetaGroup const & ) = delete;
    GradientMetaGroup(GradientMetaGroup       &&) = delete;
    GradientMetaGroup & operator=(GradientMetaGroup const & ) = delete;
    GradientMetaGroup & operator=(GradientMetaGroup       &&) = delete;

    static const GradientMeta & get(const size_t idx) {
        static GradientMetaGroup inst;
        return inst.metalist[idx];
    }

private:

    GradientMetaGroup()
      : facelist{
            // node: nothing
            // line: nothing
            // quadrilateral: 0, 1, 2, 3
            {1, 2, -1}, {2, 3, -1}, {3, 4, -1}, {4, 1, -1},
            // triangle: 4, 5, 6 
            {1, 2, -1}, {2, 3, -1}, {3, 1, -1},
            // hexahedron: 7, 8, 9, 10, 11, 12, 13, 14
            {2, 3, 5}, {6, 3, 2}, {4, 3, 6}, {5, 3, 4}, {5, 1, 2}, {2, 1, 6}, {6, 1, 4}, {4, 1, 5},
            // tetrahedron: 15, 16, 17, 18
            {3, 1, 2}, {2, 1, 4}, {4, 1, 3}, {2, 4, 3},
            // prism: 19, 20, 21, 22, 23, 24
            {5, 2, 4}, {3, 2, 5}, {4, 2, 3}, {4, 1, 5}, {5, 1, 3}, {3, 1, 4},
            // pyramid: 25, 26, 27, 28, 29, 30
            {1, 5, 2}, {2, 5, 3}, {3, 5, 4}, {4, 5, 1}, {1, 3, 4}, {3, 1, 2},
        }
      , metalist{
            // node
            {0, 0, 0, 0, nullptr},
            // line
            {1, 1, 1, 1, nullptr},
            // quadrilateral
            {2, 4, 4, 1.0/4, &facelist[0]},
            // triangle
            {3, 3, 3, 1.0/3, &facelist[4]},
            // hexadedron
            {4, 6, 8, 1.0/8, &facelist[7]},
            // tetrahedron
            {5, 4, 4, 1.0/4, &facelist[15]},
            // prism
            {6, 5, 6, 1.0/6, &facelist[19]},
            // pyramid
            {7, 5, 6, 1.0/6, &facelist[25]},
        }
    {}

    /**
     * The face lists associated with each neighboring cell for a vertex of a
     * fundamental gradient element.
     */
    index_type facelist[31][3];
    /**
     * The meta data table indexed by using cell type number.
     */
    GradientMeta metalist[NCLTYPE];

}; /* end class GradientMetaGroup */

} /* end namespace detail */

/**
 * The geometry shape of a general gradient element that is used to calculate
 * the gradient.
 */
template< size_t NDIM, size_t NEQ >
struct GradientShape {

    typedef UnstructuredBlock<NDIM> block_type;

    const GradientMeta & meta;
    const index_type icl;

    /**
     * Indices of neighboring cells.
     */
    index_type rcls[block_type::CLMFC];

    /**
     * Displacement from the self solution point to the gradient evaluation
     * points.
     */
    Vector<NDIM> idis[block_type::CLMFC];
    /**
     * Displacement from the neighboring solution points to the gradient
     * evaluation points.
     */
    Vector<NDIM> jdis[block_type::CLMFC];

    /**
     * Calculate and fill the following vectors using the tau parameter:
     *
     *  1. idis: Gradient evaluation point displacement from the self solution point.
     *  2. jdis: Gradient evaluation point displacement from the neighboring solution points.
     *
     * The GGE needs to be displaced so that the centroid colocates with the
     * solution point (i.e., centroid of the CCE).
     *
     * @param[in] block  The unstructured mesh definition.
     * @param[in] icl    The index of self cell.
     * @param[in] tau    Tau parameter (as in the c-tau scheme) of this cell.
     */
    GradientShape(
        const block_type & block
      , const LookupTable<real_type, NDIM> & cecnd
      , const index_type icl
      , const real_type tau
    )
      : meta(detail::GradientMetaGroup::get(block.cltpn()[icl])), icl(icl)
    {
        const auto & tclfcs = block.clfcs()[icl];
        const auto & icecnd = reinterpret_cast<const Vector<NDIM> &>(cecnd[icl]);

        // Calculate gradient evaluation points by using tau.
        assert(tau <= 1.0 && tau >= 0);
        const auto tau1 = 1.0 - tau;
        for (index_type ifl=0; ifl<meta.clnfc; ++ifl) {
            const auto jcl = block.fcrcl(tclfcs[ifl+1], icl);
            rcls[ifl] = jcl;
            const auto & jcecnd = reinterpret_cast<const Vector<NDIM> &>(cecnd[jcl]);
            const auto middle = (icecnd + jcecnd) / 2;
            jdis[ifl] = (middle - jcecnd) * tau1;
            idis[ifl] = jcecnd + jdis[ifl]; // At this point it's still an absolute coordinate.
        }

        // Calculate average point.
        Vector<NDIM> crd(0.0);
        for (index_type ifl=0; ifl<meta.clnfc; ++ifl) { crd += idis[ifl]; }
        crd /= meta.clnfc;
        // Triangulate the GGE into sub-GE using the average point and then
        // calculate the centroid.
        Vector<NDIM> cnd(0.0);
        real_type voc = 0;
        for (index_type isub=0; isub<meta.nsub; ++isub) {
            Vector<NDIM> subcnd(crd);
            Matrix<NDIM> dst;
            for (index_type ivx=0; ivx<NDIM; ++ivx) {
                const index_type ifl = meta.pface[isub][ivx]-1;
                subcnd += idis[ifl];
                dst[ivx] = idis[ifl] - crd;
            }
            subcnd /= NDIM+1;
            const auto vob = volume(dst);
            voc += vob;
            cnd += subcnd * vob;
        }
        cnd /= voc;

        // Shift the GGE vertices.
        const auto gsft = icecnd - cnd;
        for (index_type ifl=0; ifl<meta.clnfc; ++ifl) {
            jdis[ifl] += gsft; // Shift it so that the GGE centroid colocate with the self solution point.
            idis[ifl] -= cnd; // Make it relative to the self solution point.
        }
    }

}; /* end struct GradientShape */

/**
 * Calculate the weight of gradient and the derivative.
 */
template< size_t NDIM, size_t NEQ, int32_t ALPHA=0, bool TAYLOR=true >
struct GradientWeigh {

    typedef SolutionHouse<NDIM, NEQ> house_type;
    typedef GradientShape<NDIM, NEQ> shape_type;

    static constexpr size_t MFGE = GradientMeta::MFGE;

    static constexpr real_type ALMOST_ZERO=1.e-200;

    const shape_type & gshape;
    const house_type & shouse;

    /**
     * Gradient of each fundamental gradient element.
     */
    Vector<NDIM> grad[MFGE][NEQ];
    /**
     * Weighting of each fundamental gradient element.
     */
    real_type widv[MFGE][NEQ];
    /**
     * Total weighting of all fundamental gradient elements.
     */
    real_type wacc[NEQ];
    real_type sigma_max[NEQ]; // FIXME: document

    /**
     * @param[in] gshape  The shape of gradient elements.
     * @param[in] shouse  Container of solution variables and their derivatives.
     * @param[in] hdt     Half increment time (delta t).
     * @param[in] sgm0    Parameter for weighting.
     */
    GradientWeigh(
        const shape_type & gshape
      , const house_type & shouse
      , const real_type hdt
      , const real_type sgm0
    )
      : gshape(gshape)
      , shouse(shouse)
    {
        // calculate gradient and weighting delta.
        for (index_type ieq=0; ieq<NEQ; ++ieq) { wacc[ieq] = 0; }
        for (index_type isub=0; isub<gshape.meta.nsub; ++isub) {
            // interpolated solution
            const auto udf = interpolate_solution(gshape.meta.pface[isub], hdt);
            // displacement matrix
            const auto dst = get_displacement(gshape.meta.pface[isub]);
            // inverse (unnormalized) dosplacement matrix
            const auto dnv = unnormalized_inverse(dst);
            real_type voc;
            if (NDIM == 3) { voc = dnv.column(2).dot(dst[2]); }
            else           { voc = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0]; }
            // calculate grdient and weighting delta.
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                // store for later widv.
                grad[isub][ieq] = product(dnv, udf[ieq]) / voc;
                // W-1/2 weighting function.
                real_type wgt = grad[isub][ieq].square();
                if (0 == ALPHA) { wgt = 1.0 / sqrt(wgt+ALMOST_ZERO); }
                else            { wgt = 1.0 / pow(sqrt(wgt+ALMOST_ZERO), ALPHA); }
                // store and accumulate weighting function.
                wacc[ieq] += wgt;
                widv[isub][ieq] = wgt;
            }
        }

        // calculate W-3/4 delta and sigma_max.
        real_type wpa[NEQ][2]; // W-3/4 parameter.
        for (index_type ieq=0; ieq<NEQ; ++ieq) { wpa[ieq][0] = wpa[ieq][1] = 0.0; }
        const auto ofg1 = gshape.meta.nsub_inverse;
        for (index_type isub=0; isub<gshape.meta.nsub; isub++) {
            for (index_type ieq=0; ieq<NEQ; ieq++) {
                const real_type wgt = widv[isub][ieq] / wacc[ieq] - ofg1;
                widv[isub][ieq] = wgt;
                wpa[ieq][0] = fmax(wpa[ieq][0], wgt);
                wpa[ieq][1] = fmin(wpa[ieq][1], wgt);
            }
        }
        for (index_type ieq=0; ieq<NEQ; ++ieq) {
            sigma_max[ieq] = fmin(
                (1.0-ofg1)/(wpa[ieq][0]+ALMOST_ZERO),
                    -ofg1 /(wpa[ieq][1]-ALMOST_ZERO)
            );
            sigma_max[ieq] = fmin(sigma_max[ieq], sgm0);
        }
    }

    /**
     * @param[out] dsoln  The result derivative.
     */
    void operator() (Vector<NDIM> (&dsoln)[NEQ]) const {
        for (index_type ieq=0; ieq<NEQ; ++ieq) { dsoln[ieq] = 0; }
        const auto ofg1 = gshape.meta.nsub_inverse;
        for (index_type isub=0; isub<gshape.meta.nsub; ++isub) {
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                const real_type wgt = ofg1 + sigma_max[ieq] * widv[isub][ieq]; // FIXME: optimize
                dsoln[ieq] += wgt * grad[isub][ieq];
            }
        }
    }

private:

    Matrix<NDIM> get_displacement(const index_type (&tface)[3]) const {
        Matrix<NDIM> dst;
        for (index_type ivx=0; ivx<NDIM; ++ivx) { dst[ivx] = gshape.idis[tface[ivx]-1]; }
        return dst;
    }

    std::array<Vector<NDIM>, NEQ> interpolate_solution(
        const index_type (&tface)[3]
      , const real_type hdt
    ) const {
        std::array<Vector<NDIM>, NEQ> udf;
        const auto & tisoln = shouse.sol[gshape.icl];
        for (index_type ivx=0; ivx<NDIM; ++ivx) {
            const index_type ifl = tface[ivx]-1;
            const auto jcl = gshape.rcls[ifl];
            const auto & tjsol = shouse.sol[jcl];
            const auto & tjsoln = shouse.soln[jcl];
            const auto & tjsolt = shouse.solt[jcl];
            auto tjdsol = reinterpret_cast<const Vector<NDIM> (&)[NEQ]>(shouse.dsoln[jcl]);
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                if (TAYLOR) { udf[ieq][ivx] = tjsol[ieq] + hdt*tjsolt[ieq] - tisoln[ieq]; }
                else        { udf[ieq][ivx] = tjsoln[ieq] - tisoln[ieq]; }
                udf[ieq][ivx] += gshape.jdis[ifl].dot(tjdsol[ieq]);
            }
        }
        return udf;
    }

}; /* end struct GradientWeigh */

template< size_t NDIM >
void Solver<NDIM>::calc_dsoln() {
    // references.
    const auto & block = *m_block;
    const auto & cfl = m_sol.cfl;
    auto & dsoln = m_sol.dsoln; // output.
    const real_type hdt = m_state.time_increment * 0.5;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        // determine sigma0 and tau.
        const real_type sgm0 = m_param.sigma0 / fabs(cfl[icl]);
        const real_type tau = m_param.taumin + fabs(cfl[icl]) * m_param.tauscale;
        // calculate gradient.
        const GradientShape<NDIM,NEQ> gshape(block, m_cecnd, icl, tau);
        const GradientWeigh<NDIM,NEQ> gweigh(gshape, m_sol, hdt, sgm0);
        gweigh(reinterpret_cast<Vector<NDIM> (&)[NEQ]>(dsoln[icl]));
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
