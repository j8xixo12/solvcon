# -*- coding: UTF-8 -*-
#
# Copyright (c) 2017, Taihsiang Ho <tai271828@gmail.com>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# Description:
#   1D Sod Tube analytic solution solver.
#
#   This program is implemented by OO style to be
#   a part of ipython notebook demo materials.
#
#   The derivation of the equations for the analytic solution
#   is based on the book,
#   Principles of Computational Fluid Dynamics,
#   written by Pieter Wesseling.
#

import scipy.optimize as so
import argparse

# a number to claim two floating number value are equal.
delta_precision = 0.0000000000001


class Sod1D(object):
    """
    The core object to generate the 1D Sod tube test
    """

    def __init__(self):
        # initial condition
        # [(rhol, ul, pl), (rhor, ur, pr)]
        #
        # Sod's initial condition
        self.RHOL = 1.0
        self.UL = 0.0
        self.PL = 1.0
        self.RHOR = 0.125
        self.UR = 0.0
        self.PR = 0.1
        self.initcondition_sod = [(self.RHOL, self.UL, self.PL),
                                  (self.RHOR, self.UR, self.PR)]
        # initial condition for a shock tube problem
        # default is Sod's initial condition
        # users could change this initial conditions
        self.initcondition = self.initcondition_sod
        # constants and conventions
        self.GAMMA = 1.4  # ideal gas constant
        self.GAMMA2 = (self.GAMMA - 1.0) / (self.GAMMA + 1.0)
        self.ALPHA = (self.GAMMA + 1.0) / (self.GAMMA - 1.0)
        self.BETA = (self.GAMMA - 1.0) / (2.0 * self.GAMMA)

    def get_initcondition(self):
        return self.initcondition

    def set_initcondition(self, initcondition):
        self.initcondition = initcondition

    def set_solution(self, solution,
                     center=0.0, coor_x=0.0, time=0.0,
                     density=0.0, velocity=0.0, pressure=0.0):
        solution['center'] = center
        solution['x'] = coor_x
        solution['time'] = time
        solution['rho'] = density
        solution['v'] = velocity
        solution['p'] = pressure

    def set_solution_interface(self, solution, i12, i23, i34, i45):
        solution['I12'] = i12
        solution['I23'] = i23
        solution['I34'] = i34
        solution['I45'] = i45

    def get_analytic_solution(self, time, coor_x, center=0.0):
        """
        Get analytic solutions by giving locations (mesh) and time.

        :param coor_x float, x coordinate value.
        :param time: float, time
        :parem center: float, the coordinate of center
        :return: a list of solution. Each element is (time, rho, v, p)
        """
        location = coor_x - center

        rho4 = self.get_analytic_density_region4()
        u4 = self.get_analytic_velocity_region4()
        p4 = self.get_analytic_pressure_region4()

        rho3 = self.get_analytic_density_region3()
        u3 = self.get_analytic_velocity_region3()
        p3 = self.get_analytic_pressure_region3()

        x_shock = self.get_velocity_shock() * time
        x_disconti = u3 * time
        x_fan_right = self.get_velocity_fan_right() * time
        x_fan_left = self.get_velocity_fan_left() * time

        solution = {"rho": None, "v": None, "p": None}

        if location < x_fan_left or location == x_fan_left:
            self.set_solution(solution,
                              center, coor_x, time,
                              self.get_density_region1(),
                              self.get_velocity_region1(),
                              self.get_pressure_region1())

        elif location > x_fan_left and\
                (location < x_fan_right or location == x_fan_right):
            d = self.get_analytic_density_region2(location, time)
            v = self.get_analytic_velocity_region2(location, time)
            p = self.get_analytic_pressure_region2(location, time)
            self.set_solution(solution,
                              center, coor_x, time,
                              d, v, p)
        elif location > x_fan_right and\
                (location < x_disconti or location == x_disconti):
            self.set_solution(solution,
                              center, coor_x, time,
                              rho3, u3, p3)
        elif location > x_disconti and\
                (location < x_shock or location == x_shock):
            self.set_solution(solution,
                              center, coor_x, time,
                              rho4, u4, p4)
        elif location > x_shock:
            self.set_solution(solution,
                              center, coor_x, time,
                              self.get_density_region5(),
                              self.get_velocity_region5(),
                              self.get_pressure_region5())
        else:
            print("Something wrong!!!")

        self.set_solution_interface(solution,
                                    x_fan_left,
                                    x_fan_right,
                                    x_disconti,
                                    x_shock)

        return solution

    ##########################
    # Analytical formula
    ##########################
    def analytic_pressure_region4(self, x):
        """
        x: the root value we want to know.

        This method return the formula to get the solution
        of the pressure in the region 4.
        It is a equation that could get the solution
        by numerical approaches, e.g. Newton method.

        For details how to derive the equation, someone
        could refer to, for example, the equation (10.51)
        of Pieter Wesseling,
        Principles of Computational Fluid Dynamics

        The method and the return equation will be
        used by scipy numerial method, e.g.
        scipy.newton
        So, the method and the return value format
        follow the request of scipy.
        """
        p1 = self.PL
        p5 = self.get_pressure_region5()
        c1 = self.get_velocity_c1()
        c5 = self.get_velocity_c5()
        beta = self.BETA
        gamma = self.GAMMA
        return ((x / p1) - \
                ((1.0 - \
                  ((gamma - 1.0) * c5 * ((x / p5) - 1.0)) / \
                  (c1 * ((2.0 * gamma * (gamma - 1.0 + (gamma + 1.0) * (x / p5))) ** 0.5)) \
                  ) ** (1.0 / beta))
                )

    ################
    # Velocity
    ################
    def get_velocity_fan_left(self):
        c1 = self.get_velocity_c1()
        return -c1

    def get_velocity_fan_right(self):
        u3 = self.get_analytic_velocity_region3()
        c3 = self.get_velocity_c3()
        return u3 - c3

    def get_velocity_shock(self):
        # P409, Wesseling P.
        c5 = self.get_velocity_c5()  # 1.0583
        gamma = self.GAMMA
        p4 = self.get_analytic_pressure_region4()  # 0.3031
        p5 = self.get_pressure_region5()  # 0.1
        return c5 * ((1.0 + (((gamma + 1.0) * ((p4 / p5) - 1.0)) / (2.0 * gamma))) ** 0.5)

    def get_velocity_c1(self):
        return (self.GAMMA * self.PL / self.RHOL) ** 0.5

    def get_velocity_c3(self):
        p3 = self.get_analytic_pressure_region3()
        rho3 = self.get_analytic_density_region3()
        return (self.GAMMA * p3 / rho3) ** 0.5

    def get_velocity_c5(self):
        return (self.GAMMA * self.PR / self.RHOR) ** 0.5

    def get_velocity_region1(self):
        return self.UL

    def get_analytic_velocity_region2(self, x, t):
        c1 = self.get_velocity_c1()
        gamma = self.GAMMA
        return 2.0 / (gamma + 1.0) * (c1 + x / t)

    def get_analytic_velocity_region3(self):
        return self.get_analytic_velocity_region4()

    def get_analytic_velocity_region4(self):
        """
        The equation could be found in the
        equation next to (10.48), Wesseling P.,
        Principles of Computational Fluid Dynamics
        """
        gamma = self.GAMMA
        p4 = self.get_analytic_pressure_region4()
        p5 = self.get_pressure_region5()
        p = p4 / p5
        c5 = self.get_velocity_c5()
        return c5 * (p - 1.0) * (2.0 / (gamma * (gamma - 1.0 + (gamma + 1.0) * p))) ** 0.5

    def get_velocity_region5(self):
        return self.UR

    ################
    # Pressure
    ################
    def get_pressure_region1(self):
        return self.PL

    def get_analytic_pressure_region2(self, x, t):
        # (10.44) Wesssling P.
        c1 = self.get_velocity_c1()
        u2 = self.get_analytic_velocity_region2(x, t)
        p1 = self.PL
        gamma = self.GAMMA
        beta = self.BETA
        return p1 * (1.0 - (gamma - 1.0) * u2 / 2 / c1) ** (1.0 / beta)

    def get_analytic_pressure_region3(self):
        return self.get_analytic_pressure_region4()

    def get_analytic_pressure_region4(self):
        return self.get_analytic_pressure_region4_by_newton()

    def get_analytic_pressure_region4_by_newton(self, x0=1):
        """
        x0 : the guess initial value to be applied in Newton method
        """
        return so.newton(self.analytic_pressure_region4, x0)

    def get_pressure_region5(self):
        return self.PR

    ################
    # Density
    ################
    def get_density_region1(self):
        return self.RHOL

    def get_analytic_density_region2(self, x, t):
        # (10.45), Wesseling P.
        # Principles of Computational Fluid Dynamics
        gamma = self.GAMMA
        rho1 = self.RHOL
        p1 = self.get_pressure_region1()
        p2 = self.get_analytic_pressure_region2(x, t)
        return rho1 * (p2 / p1) ** (1.0 / gamma)

    def get_analytic_density_region3(self):
        # P410, Wesseling P.
        # Principles of Computational Fluid Dynamics
        rho1 = self.get_density_region1()
        p1 = self.get_pressure_region1()
        p3 = self.get_analytic_pressure_region3()
        return rho1 * (p3 / p1) ** (1.0 / self.GAMMA)

    def get_analytic_density_region4(self):
        # P410, Wesseling P.
        # Principles of Computational Fluid Dynamics
        alpha = self.ALPHA
        p4 = self.get_analytic_pressure_region4()
        p5 = self.get_pressure_region5()
        p = p4 / p5
        rho5 = self.get_density_region5()
        return rho5 * (1.0 + alpha * p) / (alpha + p)

    def get_density_region5(self):
        return self.RHOR


def get_solution(time, coor_x, coor_center=0.0):
    """
    Get analytic solution.

    This method returns not only the values of the usual physics quantities,
    but also the location of regions. The region information is useful for
    users to know where the discontinutity is.

    :param t: float, time
    :param location: float, location. In this 1D case, x coordinate value.
    :param tube_length: float
    :param center_location: flaot
    :return: solution object, I12 mean interface between region 1 and 2 etc.

             {
             "center": float coordinate,
             "x": float coordinate,
             "time": float t,
             "rho": float rho,
             "v": float v,
             "p": float p,
             "I12": float coordinate,
             "I23": float coordinate,
             "I34": float coordinate,
             "I45": float coordinate
             }

             I12 is actually fan_left location
             I23 is actually fan_right location
             I34 is actually x_discontinutity location
             I45 is actually x_shock location
    """
    sod = Sod1D()
    return sod.get_analytic_solution(time, coor_x, coor_center)


if __name__ == '__main__':
    # TODO: handling args and delegate the algorithm to get_solution
    # arguments
    #   general: time, location center_location
    #   level: simple (physics quantity only)
    #          default (physics quantity and interface location)
    #   format: default (tuple for simple and dict for default lovel)
    #           json
    # return get_solution(time, location, center_location)
    parser = argparse.ArgumentParser()
    parser.add_argument('time', type=float,
                        help="The value of the time.")
    parser.add_argument('coor_x', type=float,
                        help="The value of x coordinate.")
    parser.add_argument('-c', '--center', type=float, default=0.0, required=False,
                        help="The value of the center coordinate. The default is 0.")

    args = parser.parse_args()

    print(get_solution(args.time, args.coor_x, args.center))
