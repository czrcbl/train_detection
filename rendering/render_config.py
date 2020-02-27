import os
import math

# add initial rotation angle for parts.
parts = [
            {
            'name': 'tray',
            'rot': (0, 0, 0)
            },
            {
            'name': 'dosing_nozzle', 
            'rot': (math.pi/2, 0, 0)
            },
            {
            'name': 'button_pad', 
            'rot': (0, 0, 0)
            },
            {
            'name': 'part1', 
            'rot': (-math.pi/2, 0, 0)
 
            },
                        {
            'name': 'part2', 
            'rot': (math.pi/2, 0, 0)

            },
            {
            'name': 'part3',
            'rot': (math.pi/2, 0, 0)
            }
        ]


params = {
    'test': {
        'vert_angles': [math.pi/6][:1],
        'rot_angles': [0],
        'dists': [300]
    },
    'small': {
        'vert_angles': (0, math.pi/6, math.pi/3, -math.pi/3, -math.pi/6),
        'rot_angles': [math.pi/2/5 * x for x in range(5)],
        'dists': [300, 600, 1200]
    },
    'large': {
        'vert_angles':  (0, math.pi/6, math.pi/3, math.pi/2, -math.pi/3, -math.pi/6),
        'rot_angles': [math.pi/2/8 * x for x in range(8)],
        'dists': [300, 600, 900, 1200, 1500, 1800]
    }
}
# parts = [
#             {
#             'name': 'tray',
#             'rot': (0, 0, 0),
#             'dists': [300, 600, 1200]
#             },
#             {
#             'name': 'dosing_nozzle', 
#             'rot': (math.pi/2, 0, 0),
#             'dists': [200, 400, 800]
#             },
#             {
#             'name': 'button_pad', 
#             'rot': (0, 0, 0),
#             'dists': [600, 1200, 1800]
#             },
#             {
#             'name': 'part1', 
#             'rot': (-math.pi/2, 0, 0),
#             'dists': [150, 300, 600]
#             },
#                         {
#             'name': 'part2', 
#             'rot': (math.pi/2, 0, 0),
#             'dists': [150, 300, 600]
#             },
#             {
#             'name': 'part3',
#             'rot': (math.pi/2, 0, 0),
#             'dists': [150, 300, 600]
#             }
#         ]