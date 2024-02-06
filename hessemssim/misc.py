"""This module gathers some loose functions that are used at various points
within the toolbox from various modules, functions or classes.

This module contains functions implementing data structure manipulation and
similar things. I.e., it does not implement
mathematical formula to calculate something specific to the scientific
domain. See module 'util' for this kind of functions."""

import numpy as np


def separate_tuples(arraytuple, index, delay=1):
    """Takes an input of an array of tuples and iterates over the array to
    return an array with only the first or second entries of every tuple.
    The iteration will begin after delay number of steps to include the
    option to ignore starting values"""
    arraysingle = np.array([ele[index] for ele in arraytuple[delay:]
                            if np.size(ele) == 2])
    return arraysingle


def create_colorscheme():
    """Stores the colorscheme colors turquoise - violet - ochre for base,
    peak and hybrid storage as well as blue - orange - green as
    1st/2nd/3rd highlight color.
    See as a source
    http://paletton.com/#uid=33i0X0kllllaFw0g0qFqFg0w0aF and
    http://paletton.com/#uid=34e0X0krgoNfW-LlQsLuwjfF-ci """
    c = dict(turquoise='#226666', turquoise1='#669999', turquoise2='#407F7F',
             turquoise3='#226666', turquoise4='#0D4D4D', turquoise5='#003333',
             violet='#882D61', violet1='#CD88AF', violet2='#AA5585',
             violet3='#882D61', violet4='#661141', violet5='#440027',
             ochre='#AA9739', ochre1='#FFF0AA', ochre2='#D4C26A',
             ochre3='#AA9739', ochre4='#806D15', ochre5='#554600',
             blue='#3A2188', blue1='#7C69BB', blue2='#553E9E',
             blue3='#3A2188', blue4='#26106A', blue5='#140543',
             orange='#C66A1D', orange1='#FFBA80', orange2='#E59049',
             orange3='#C66A1D', orange4='#994907', orange5='#622C00',
             green='#84B81B', green1='#C9F179', green2='#A5D644',
             green3='#84B81B', green4='#628F07', green5='#3D5B00',
             white='#FFFFFF', black='#000000',
             grey='#A1A1A1', grey1='#D7D7D7', grey2='#B7B7B7',
             grey3='#A1A1A1', grey4='#505050', grey5='#212121')

    # create alias names for colors
    orig = ['turquoise', 'violet', 'ochre', 'blue', 'orange', 'green']
    alias = ['base', 'peak', 'hybrid', '1st', '2nd', '3rd']
    number = ['', '1', '2', '3', '4', '5']
    for o, a in zip(orig, alias):
        for n in number:
            c[a + n] = c[o + n]

    return c


COLORS = create_colorscheme()
