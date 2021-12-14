import numpy as np

# if Classification - Class 6
# merge_category_dict = {
#     1 : [1, 2],
#     2 : [3, 7, 8, 10, 11],
#     3 : [4],
#     4 : [5],
#     5 : [6],
#     6 : [9],
# }
#
# merge_category_color = {
#     1: (0, 255, 0), # green # Human
#     2: (255, 0, 0), # blue # bicycle
#     3: (0, 0, 255), # red # car
#     4: (0, 255, 255), # yellow # van
#     5: (128, 0, 128), # violet # truck
#     6: (0, 165, 255), # orange # bus
# }

# if Classification - Class 3
# merge_category_dict = {
#     1 : [1, 2],
#     2 : [3, 7, 8, 10, 11],
#     3 : [4, 5, 6, 9],
# }
#
# merge_category_color = {
#     1: (0, 255, 0), # green # Human
#     2: (255, 0, 0), # blue # bicycle
#     3: (0, 0, 255), # violet # car
# }
# kpt_oks_sigmas = np.array([0.15, 0.15, 0.15])/10.0

# if not Classification
merge_category_dict = {
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

merge_category_color = {
    1: (0, 0, 255),
}

kpt_oks_sigmas = np.array([0.15])/10.0