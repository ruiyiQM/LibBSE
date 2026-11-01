#!/usr/bin/env python3
"""Generate FHI-aims band segments covering an unshifted cubic k mesh."""
# Author: Ruiyi Zhou.
import argparse


def band_lines(nk):
    if nk < 2:
        raise ValueError('grid size must be at least 2')
    for i in range(nk):
        for j in range(nk):
            x, y = i / nk, j / nk
            yield (f'output band {x:.16f} {y:.16f} 0.0000000000000000 '
                   f'{x:.16f} {y:.16f} {(nk - 1) / nk:.16f} {nk}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nk', type=int, nargs='?', default=3,
                        help='cubic grid size (default: 3)')
    args = parser.parse_args()
    if args.nk < 2:
        parser.error('grid size must be at least 2')
    print('\n'.join(band_lines(args.nk)))


if __name__ == '__main__':
    main()
