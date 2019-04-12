import cProfile
import pstats
import random

from AgraphExample import main

random.seed(0)

def profile_and_print_stats(routine, sortby="time"):
    profiler = cProfile.Profile()
    profiler.enable()
    routine()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(sortby)
    stats.print_stats()

if __name__ == '__main__':
    main()
    