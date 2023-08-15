import numpy as np
'''
Constants used by other methods and classes.

sectors : list
    List of list of tuples of the angle boundaries which enclose the sectors and a list of the edge cases,
    i.e. diagonal lines.

direction_dict : dict
    Dictonary numbering the sectors and their corresponding primary and secondary direction.

slope_dict : dict
    Dictionary for each direction and if its slope, e.g. "u>" means up and positive (>0) slope.

ci_j : float
    Are the coefficients for the hermite polynomials, where i is the degree and j is the number of coefficient
    counting from left to right.
'''
sectors = [[(45, 90), (0, 45), (-45, 0),
            (-90, -45)], [90, 45, 0, -45, -90]]

direction_dict = {0 : ("lu","rd"), 1 : ("ul","dr"),
                  2 : ("ur","dl"), 3 : ("ru","ld"),
                  4 : ("l ","r "), 5 : ("ul","dr"),
                  6 : ("u ","d "), 7 : ("ur","dl"),
                  8 : ("r ","l ")}

slope_dict = { "u>" : (-1, 1), "u<" : ( 1, 1),
               "d>" : ( 1,-1), "d<" : (-1,-1),
               "r>" : ( 1,-1), "r<" : ( 1, 1),
               "l>" : (-1, 1), "l<" : (-1,-1),
               "l=" : (-1, 0), "r=" : ( 1, 0),
               "u=" : ( 0, 1), "d=" : ( 0,-1)}

# coefficients for h2
c2_1 = np.sqrt(2)
c2_2 = np.sqrt(2) / 2

# coefficients for h4
c4_1 = 4  / np.sqrt(24)
c4_2 = 12 / np.sqrt(24)
c4_3 = 3  / np.sqrt(24)
