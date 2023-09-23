from os import environ
from os.path import dirname, join, realpath
import subprocess
import sys

if __name__ == '__main__':
    dir = dirname(__file__)
    requirements = join(dir, 'requirements', 'cpu.txt')

    if environ.get('CUDA_PATH_V11_7') is not None:
        requirements = join(dir, 'requirements', 'cu117.txt')

    if environ.get('CUDA_PATH_V11_8') is not None:
        requirements = join(dir, 'requirements', 'cu118.txt')

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-r', realpath(requirements)])
