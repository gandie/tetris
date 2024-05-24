import io
import pickle
import numpy as np
import torch
import logging

from datetime import datetime
from core.gen_algo import get_score, Population
from core.utils import check_needed_turn, do_action, drop_down, \
    do_sideway, do_turn, check_needed_dirs, feature_names
from pyboy import PyBoy
from multiprocessing import Pool, cpu_count

import argparse
import signal

logger = logging.getLogger("tetris")
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs.out')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def eval_network(epoch, child_index, child_model, run_per_child, max_score, show):

    window = "SDL2" if show else "null"
    pyboy = PyBoy('tetris_1.1.gb', window=window)
    pyboy.set_emulation_speed(0)
    tetris = pyboy.game_wrapper
    tetris.start_game()

    # Set block animation to fall instantly
    pyboy.memory[0xff9a] = 2

    run = 0
    actions = 0
    scores = []
    levels = []
    lines = []

    while run < run_per_child:
        # Beginning of action
        best_action_score = np.NINF
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        pyboy.save_state(begin_state)
        # Number of lines at the start
        s_lines = tetris.lines

        # Determine how many possible rotations we need to check for the block
        block_tile = pyboy.memory[0xc203]
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', pyboy, n_dir=1,
                                  n_turn=turns_needed):
            score = get_score(tetris, child_model, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do left
        for move_dir in do_action('Left', pyboy, n_dir=lefts_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris, child_model, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do right
        for move_dir in do_action('Right', pyboy, n_dir=rights_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris, child_model, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do best action
        for _ in range(best_action['Turn']):
            do_turn(pyboy)
        for _ in range(best_action['Left']):
            do_sideway(pyboy, 'Left')
        for _ in range(best_action['Right']):
            do_sideway(pyboy, 'Right')
        drop_down(pyboy)
        pyboy.tick()
        actions += 1

        # Game over:
        if tetris.game_over() or tetris.score == max_score or actions > 100:
            scores.append(tetris.score)
            levels.append(tetris.level)
            lines.append(tetris.lines)
            if run == run_per_child - 1:
                pyboy.stop()
            else:
                actions = 0
                tetris.reset_game()
            run += 1

    child_fitness = np.average(scores)
    logger.info("-" * 20)
    logger.info("Iteration %s - child %s" % (epoch, child_index))
    logger.info("Score: %s, Level: %s, Lines %s" % (scores, levels, lines))
    logger.info("Fitness: %s" % child_fitness)
    logger.info("Output weight:")
    weights = {}
    for i, j in zip(feature_names, child_model.output.weight.data.tolist()[0]):
        weights[i] = np.round(j, 3)
    logger.info(weights)

    return child_fitness


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(n_workers, pop_size, epochs, run_per_child, max_score, show):
    e = 0
    p = Pool(n_workers, initializer=initializer)
    population = None
    max_fitness = 0

    while e < epochs:
        start_time = datetime.now()
        if population is None:
            if e == 0:
                population = Population(size=pop_size)
            else:
                with open('checkpoint/checkpoint-%s.pkl' % (e - 1), 'rb') as f:
                    population = pickle.load(f)
        else:
            population = Population(size=pop_size, old_population=population)

        result = [0] * pop_size
        for i in range(pop_size):
            result[i] = p.apply_async(
                eval_network,
                (e, i, population.models[i], run_per_child, max_score, show)
            )

        for i in range(pop_size):
            try:
                population.fitnesses[i] = result[i].get()
            except KeyboardInterrupt:
                logger.error("KeyboardInterrupt, stopping ...")
                p.terminate()
                return

        logger.info("-" * 20)
        logger.info("Iteration %s fitnesses %s" % (
            e, np.round(population.fitnesses, 2)))
        logger.info(
            "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
        logger.info(
            "Iteration %s mean fitness %s " % (e, np.mean(
                population.fitnesses)))
        logger.info("Time took %s" % (datetime.now() - start_time))
        logger.info("Best child output weights:")
        weights = {}
        for i, j in zip(feature_names, population.models[np.argmax(
                population.fitnesses)].output.weight.data.tolist()[0]):
            weights[i] = np.round(j, 3)
        logger.info(weights)
        # Saving population
        with open('checkpoint/checkpoint-%s.pkl' % e, 'wb') as f:
            pickle.dump(population, f)

        if np.max(population.fitnesses) >= max_fitness:
            max_fitness = np.max(population.fitnesses)
            file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
                np.round(max_fitness, 2))
            # Saving best model
            torch.save(
                population.models[np.argmax(
                    population.fitnesses)].state_dict(),
                'models/%s' % file_name)
        e += 1

    p.close()
    p.join()


def cli_args():
    parser = argparse.ArgumentParser("Tetris training")
    parser.add_argument(
        '-e',
        '--epochs',
        help="How many epochs to train",
        type=int,
        default=2,
    )
    parser.add_argument(
        '-r',
        '--runs',
        help="Tetris games played per child",
        type=int,
        default=3,
    )
    parser.add_argument(
        '-p',
        '--popsize',
        help="Size of population",
        type=int,
        default=50,
    )
    parser.add_argument(
        '-m',
        '--maxscore',
        help="Maximum score in one run, game is stopped if reached",
        type=int,
        default=999999,
    )
    parser.add_argument(
        '-n',
        '--numworkers',
        help="Number of workers (pyboys) to spawn",
        type=int,
        default=cpu_count(),
    )
    parser.add_argument(
        '-s',
        '--show',
        help="Do not run pyboys in headless mode, render each one in window",
        default=False,
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = cli_args()
    main(
        n_workers=args.numworkers,
        pop_size=args.popsize,
        epochs=args.epochs,
        run_per_child=args.runs,
        max_score=args.maxscore,
        show=args.show,
    )
