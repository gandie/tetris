import io
import neat
import numpy as np
import os
import pickle
import visualize

from core.gen_algo import get_score
from core.utils import do_best_action, spawn_pyboy
from pyboy import PyBoy


epochs = 4
max_fitness = 0
max_score = 999999
n_workers = 16
actions_limit = 100

def eval_genome(genome, config, show=False):
    global max_fitness

    pyboy, tetris = spawn_pyboy(show=show)

    # Set block animation to fall instantly
    pyboy.memory[0xff9a] = 2

    model = neat.nn.FeedForwardNetwork.create(genome, config)
    child_fitness = 0
    actions = 0

    while True:

        best_action = do_best_action(get_score, pyboy, tetris, model, neat=True)

        actions += 1

        game_over = tetris.game_over()
        got_highscore = tetris.score == max_score
        actions_depleted = actions > actions_limit

        # Game over:
        if game_over or got_highscore or actions_depleted:
            child_fitness = tetris.score
            if tetris.score == max_score:
                print("Max score reached")
            # punish loosing
            if game_over:
                child_fitness = 0
            break

    # Dump best model
    if child_fitness >= max_fitness and child_fitness > 1000:
        max_fitness = child_fitness
        file_name = str(np.round(max_fitness, 2))
        with open('neat_models/%s' % file_name, 'wb') as f:
            pickle.dump(model, f)
        with open('neat_models/%s_genome' % file_name, 'wb') as f:
            pickle.dump(genome, f)

    pyboy.stop()
    return child_fitness


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    # Uncomment to load from checkpoint
    # p = neat.Checkpointer().restore_checkpoint('checkpoint/neat-checkpoint-62')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix='checkpoint/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(n_workers, eval_genome)
    winner = p.run(pe.evaluate, epochs)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = {-1: 'agg_height', -2: 'n_holes', -3: 'bumpiness',
                  -4: 'cleared', -5: 'num_pits', -6: 'max_wells',
                  -7: 'n_cols_with_holes', -8: 'row_transitions',
                  -9: 'col_transitions', 0: 'Score'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')
    run(config_path)
