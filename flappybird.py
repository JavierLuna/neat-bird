import argparse
import sys

import neat
from numpy.random import choice

try:
    import cPickle as pickle
except ImportError:
    import pickle

import pygame
import pygame.locals

FPS = 200
SCREENWIDTH = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE = 160  # gap between upper and lower part of pipe

BACKGROUND = pygame.image.load('assets/background.png')


class AIBird(pygame.sprite.Sprite):
    asset = 'assets/redbird.png'

    def __init__(self, display_screen, genome, net):

        pygame.sprite.Sprite.__init__(self)
        self.score = 0
        self.net = net
        self.genome = genome
        self.image = pygame.image.load(self.asset)

        self.x = int(SCREENWIDTH * 0.2)
        self.y = SCREENHEIGHT * 0.5

        self.rect = self.image.get_rect()
        self.height = self.rect.height
        self.screen = display_screen

        self.playerVelY = -9
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        self.display(self.x, self.y)

    def display(self, x, y):

        self.screen.blit(self.image, (x, y))
        self.rect.x, self.rect.y = x, y

    def activate(self, neural_input):
        movement = self.net.activate([self.y] + neural_input)[0]
        if movement >= 0.5:
            return "UP"
        else:
            return None

    def move(self, movement_command):

        if movement_command is not None:
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

        self.y += min(self.playerVelY, SCREENHEIGHT - self.y - self.height)
        self.y = max(self.y, 0)
        self.display(self.x, self.y)


class HumanControlledBird(pygame.sprite.Sprite):
    asset = 'assets/yellowbird.png'

    def __init__(self, display_screen, genome, net):

        pygame.sprite.Sprite.__init__(self)
        self.score = 0
        self.net = net
        self.genome = genome
        self.image = pygame.image.load(self.asset)

        self.x = int(SCREENWIDTH * 0.2)
        self.y = SCREENHEIGHT * 0.5

        self.rect = self.image.get_rect()
        self.height = self.rect.height
        self.screen = display_screen

        self.playerVelY = -9
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        self.display(self.x, self.y)

    def display(self, x, y):

        self.screen.blit(self.image, (x, y))
        self.rect.x, self.rect.y = x, y

    def move(self, movement_command):

        if movement_command is not None:
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

        self.y += min(self.playerVelY, SCREENHEIGHT - self.y - self.height)
        self.y = max(self.y, 0)
        self.display(self.x, self.y)


class PipeBlock(pygame.sprite.Sprite):

    def __init__(self, image, upper):

        pygame.sprite.Sprite.__init__(self)

        if not upper:
            self.image = pygame.image.load(image)
        else:
            self.image = pygame.transform.rotate(pygame.image.load(image), 180)

        self.rect = self.image.get_rect()


class Pipe(pygame.sprite.Sprite):

    def __init__(self, screen, x):
        pygame.sprite.Sprite.__init__(self)

        self.screen = screen
        self.lowerBlock = PipeBlock('assets/pipe-red.png', False)
        self.upperBlock = PipeBlock('assets/pipe-red.png', True)

        self.pipeWidth = self.upperBlock.rect.width
        self.x = x

        heights = self.getHeight()
        self.upperY, self.lowerY = heights[0], heights[1]

        self.behindBird = 0
        self.display()

    def getHeight(self):
        rand_val = choice([1, 2, 3, 4, 5, 6, 7, 8, 9],
                          p=[0.04, 0.04 * 2, 0.04 * 3, 0.04 * 4, 0.04 * 5, 0.04 * 4, 0.04 * 3, 0.04 * 2, 0.04])

        mid_y_pos = 106 + 30 * rand_val

        upper_pos = mid_y_pos - (PIPEGAPSIZE / 2)
        lower_pos = mid_y_pos + (PIPEGAPSIZE / 2)

        return [upper_pos, lower_pos]

    def display(self):
        self.screen.blit(self.lowerBlock.image, (self.x, self.lowerY))
        self.screen.blit(self.upperBlock.image, (self.x, self.upperY - self.upperBlock.rect.height))
        self.upperBlock.rect.x, self.upperBlock.rect.y = self.x, (self.upperY - self.upperBlock.rect.height)
        self.lowerBlock.rect.x, self.lowerBlock.rect.y = self.x, self.lowerY

    def move(self):
        self.x -= 3

        if self.x <= 0:
            self.x = SCREENWIDTH
            heights = self.getHeight()
            self.upperY, self.lowerY = heights[0], heights[1]
            self.behindBird = 0

        self.display()
        return [self.x + (self.pipeWidth / 2), self.upperY, self.lowerY]


def eval_genomes_concurrent(genomes, config):
    pygame.init()

    # Initialize Pygame stuff
    FPS_CLOCK = pygame.time.Clock()
    DISPLAY = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

    # Create them pipes
    pipe1 = Pipe(DISPLAY, SCREENWIDTH + 100)
    pipe2 = Pipe(DISPLAY, SCREENWIDTH + 110 + (SCREENWIDTH / 2))

    pipe_group = pygame.sprite.Group()
    pipe_group.add(pipe1.upperBlock)
    pipe_group.add(pipe2.upperBlock)
    pipe_group.add(pipe1.lowerBlock)
    pipe_group.add(pipe2.lowerBlock)

    # Create AIBird "agents" to play the game. Each agent corresponds to a genome, which is a neural net.
    alive_birds = []
    for genome_id, genome in genomes:
        alive_birds.append(AIBird(DISPLAY, genome, neat.nn.FeedForwardNetwork.create(genome, config)))

    max_score = 0
    pygame.display.set_caption(f'Max score: {max_score}')
    while True:

        DISPLAY.blit(BACKGROUND, (0, 0))

        # Calculate some information which the agent might consider interesting.
        if (pipe1.x < pipe2.x and pipe1.behindBird == 0) or (pipe2.x < pipe1.x and pipe2.behindBird == 1):
            neural_input = [pipe1.x, pipe1.upperY, pipe1.lowerY]
        else:
            neural_input = [pipe2.x, pipe2.upperY, pipe2.lowerY]

        # The pipes gotta move, yo!
        pipe1_pos = pipe1.move()
        pipe2_pos = pipe2.move()

        for bird in alive_birds: # Only evaluate alive birds, no point if the bird is dead. Dead birds don't tend to move.
            # Feed the interesting data to the neural net.
            # It will figure out which movement (Go up or do nothing) to perform base on the given data.
            movement = bird.activate(neural_input)

            bird.move(movement) # Apply movement to the birdie. Fly! or don't, I'm not a cop.

            # Check if our beloved birdie has collided with a pipe.
            bird_collided = pygame.sprite.spritecollideany(bird, pipe_group) is not None

            if bird_collided: # Oh shit he did.
                bird.kill() # Put him out of his misery
                alive_birds.remove(bird) # Press F to pay respects
                # Assign the score to the genome's fitness so the NEAT algorithm can do it's thing.
                bird.genome.fitness = bird.score

            else: # He cool bruh! Let him be!
                # Check if he has surpassed a pipe. Give that good birdie a point :)
                if pipe1_pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2) and pipe1.behindBird == 0:
                    pipe1.behindBird = 1
                    bird.score += 1
                if pipe2_pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2) and pipe2.behindBird == 0:
                    pipe2.behindBird = 1
                    bird.score += 1

                if bird.score > max_score:
                    max_score = bird.score
                    pygame.display.set_caption(f'Max score: {max_score}')

        if not alive_birds: # All birdies in this generation died. Tell NEAT just that.
            break

        pygame.display.update()
        FPS_CLOCK.tick(FPS)

def play_vs_game(genome, config):
    pygame.init()

    FPS_CLOCK = pygame.time.Clock()
    DISPLAY = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

    pipe1 = Pipe(DISPLAY, SCREENWIDTH + 100)
    pipe2 = Pipe(DISPLAY, SCREENWIDTH + 110 + (SCREENWIDTH / 2))

    pipe_group = pygame.sprite.Group()
    pipe_group.add(pipe1.upperBlock)
    pipe_group.add(pipe2.upperBlock)
    pipe_group.add(pipe1.lowerBlock)
    pipe_group.add(pipe2.lowerBlock)
    ia_bird = AIBird(DISPLAY, genome, neat.nn.FeedForwardNetwork.create(genome, config))
    human_bird = HumanControlledBird(DISPLAY, genome, neat.nn.FeedForwardNetwork.create(genome, config))
    alive_birds = [ia_bird, human_bird]
    human_bird.update()
    max_score = 0
    pygame.display.set_caption(f'Max score: {max_score}')
    pygame.display.update()
    while True:

        DISPLAY.blit(BACKGROUND, (0, 0))

        if (pipe1.x < pipe2.x and pipe1.behindBird == 0) or (pipe2.x < pipe1.x and pipe2.behindBird == 1):
            neural_input = [pipe1.x, pipe1.upperY, pipe1.lowerY]
        else:
            neural_input = [pipe2.x, pipe2.upperY, pipe2.lowerY]

        pipe1_pos = pipe1.move()
        pipe2_pos = pipe2.move()

        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT or (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_SPACE:
                human_bird.move("UP")

        ia_bird.activate(neural_input)

        for bird in alive_birds:
            bird_collided = pygame.sprite.spritecollideany(bird, pipe_group) is not None
            if bird_collided:
                bird.kill()
            else:
                if pipe1_pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2) and pipe1.behindBird == 0:
                    pipe1.behindBird = 1
                    max_score += 1
                if pipe2_pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2) and pipe2.behindBird == 0:
                    pipe2.behindBird = 1
                    max_score += 1
            pygame.display.set_caption(f'Max score: {max_score}')
        if not alive_birds:
            break
        pygame.display.update()
        FPS_CLOCK.tick(FPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'play'))
    parser.add_argument('--genome_path', help='Path to save/load a genome.', default=None, required=True)
    parser.add_argument('--config', help='Path to the NEAT training config file', default='neat-config')
    parser.add_argument('--mihto', help="Play with Mihto skin", default=False, action='store_true')

    args = parser.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)

    if args.mihto:
        AIBird.asset = 'assets/mihtobird.png'

    if args.command == 'train':
        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        winner = pop.run(eval_genomes_concurrent)

        print(winner)

        if args.genome_path:
            with open(args.genome_path, 'wb') as output_file:
                pickle.dump(winner, output_file)
    else:
        winner = None
        if args.genome_path:
            with open(args.genome_path, 'rb') as input_file:
                winner = pickle.load(input_file)
        play_vs_game(winner, config)
