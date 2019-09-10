import os
import math

parts = [
            {
            'name': 'tray',
            'rot': (0, 0, 0),
            'dists': [300, 600, 1200]
            },
            {
            'name': 'dosing_nozzle', 
            'rot': (math.pi/2, 0, 0),
            'dists': [200, 400, 800]
            },
            {
            'name': 'button_pad', 
            'rot': (0, 0, 0),
            'dists': [600, 1200, 1800]
            },
            {
            'name': 'part1', 
            'rot': (-math.pi/2, 0, 0),
            'dists': [150, 300, 600]
            },
                        {
            'name': 'part2', 
            'rot': (math.pi/2, 0, 0),
            'dists': [150, 300, 600]
            },
            {
            'name': 'part3',
            'rot': (math.pi/2, 0, 0),
            'dists': [150, 300, 600]
            }
        ]