'''
How to run:

Add the 'NEAT_conf.txt' file the same directory as main.py and run 


'''



import pygame
import random
import time
import os
import neat
import math

gap = 200

class Pipes:
    
    def __init__(self, x):
        self.width = 100
        self.x = x
        self.speed = 10
        self.location = random.choice(range(20, (display_height-gap-20)))
        self.pssd = False
        self.color = (37, 184, 42)
        
    def create(self):
        
        pygame.draw.rect(game_display, self.color, [self.x, 0, self.width, self.location])
        pygame.draw.rect(game_display, self.color, [self.x, self.location+gap, self.width, display_height-(self.location+self.width)])
        self.x -= int(self.speed)
        
        
    def off_screen(self):
        
        if self.x+100 < 0 :
            return True
        
        return False
    
    
    def collide(self, bird):
        return (bird.x> self.x and bird.x < self.x+self.width) and (bird.y <= self.location or  bird.y >= self.location+gap)
    
    
    def passed(self, bird):
        if not self.pssd:
            
            if (bird.x > self.x+self.width) and (bird.y>self.location and bird.y < self.location+gap):
                self.pssd = True
                return True
                
        return False
        
    
        
class Bird:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 8
        self.gravity = 0.9
        self.resistance = 0.9
        self.upforce = 10
        self.is_dead = False
        
                
    def create(self):
        pygame.draw.circle(game_display, (246, 250, 7), (self.x, int(self.y)), 10)
        self.y += self.speed
        self.speed += self.gravity
        self.speed *= self.resistance
        

            

    def flap(self):
        self.speed -= self.upforce






def distance(p1, p2):
    return int(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))


## display initialization
display_width, display_height = 700, 750 
game_display=pygame.display.set_mode((display_width, display_height))

## font initialization
pygame.font.init()

myfont = pygame.font.SysFont('arial', 50)
myfont1 = pygame.font.SysFont('arial', 30)
generation = -1

def main(genomes, config):
    global score
    global generation
    score = 0
    generation += 1
    clock = pygame.time.Clock()
    pipes = []
    pipes.append(Pipes(display_width))
    game_loop = True
    tick = 0

    
    
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(40, 350))
        ge.append(genome)
        
            
    while game_loop:
    
        clock.tick(30)
        game_display.fill((29, 150, 207))
        
    #___________________________________________________
    
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                game_loop = False
                pygame.quit()
                quit()      
    #___________________________________________________  
    
        tick += 1
        
        pipe_ind = 0
        if len(birds)>0:
            if len(pipes)>1:
                if abs(birds[0].x-pipes[0].x) > abs(birds[0].x-pipes[1].x):
                    pipe_ind = 1
            
        pipes[pipe_ind].color = (200, 1, 0)
        
        if len(birds) <= 0:
            game_loop = False
            break
                
        
        for key, bird in enumerate(birds):
            ge[key].fitness += 0.3
                                  
            output = nets[key].activate( (bird.y, distance([bird.x, bird.y], [pipes[pipe_ind].x, pipes[pipe_ind].location]), distance([bird.x, bird.y], [pipes[pipe_ind].x, (pipes[pipe_ind].location+gap)])) )
              
            if output[0]>0.5:
                bird.flap()
        
        
        
        
        
        ## creating the birds and checking if it goes off-screan
        for key, bird in enumerate(birds):
            bird.create()
        
            if bird.y>display_height or bird.y<0:
                ## removing fitness for going over or below screen
                ge[key].fitness += -1
                birds.pop(key)
                nets.pop(key)
                ge.pop(key)
                

        ## creating pipes                                
        for pipe in pipes:
           
            pipe.create()
            
            if pipe.off_screen():
                pipes.remove(pipe)
            

            ## checking bird collision with pipes
            for key, bird in enumerate(birds):
                
                if pipe.collide(bird):
                    ## removing fitness for collision
                    ge[key].fitness += -2
                    birds.pop(key)
                    nets.pop(key)
                    ge.pop(key)
                
                
                if pipe.passed(bird) :
                    score+=1
                    ## rewarding for passing pipe
                    ge[key].fitness+=5
                    
                ## creating lines from birds to the pipes
                pygame.draw.line(game_display, (255, 0, 0), (bird.x, bird.y), (pipes[pipe_ind].x, pipes[pipe_ind].location), 1) 
                pygame.draw.line(game_display, (255, 0, 0), (bird.x, bird.y), (pipes[pipe_ind].x, pipes[pipe_ind].location+gap), 1) 
                    

        ## adding pipes every few seconds
        if tick%60 == 0:
            pipes.append(Pipes(display_width))
                        
                            
            
        ## adding score, gen and alive numbers on screen 
        text_score = myfont.render(str(score), False, (0, 0, 0))
        game_display.blit(text_score,(display_width/2-10,0))
        
        text_birds = myfont1.render('Alive: '+str(len(birds)), False, (255, 255, 255))
        game_display.blit(text_birds,(10, 10))
    
        text_gen = myfont1.render('Gen: '+str(generation), False, (255, 255, 255))
        game_display.blit(text_gen,(10, 40))
        
        pygame.display.update()
        







## NEAT 
def run(path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main, 100)
    
    print('\nBest genome:\n{!s}'.format(winner))
    print('Final Score: '+score)



if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'NEAT_conf.txt')
    run(config_path)
    
    
    
    
    