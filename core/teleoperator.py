
#############################################################################
# DEVELOPED BY KANISHK                                          #############
# THIS SCRIPT CONSTRUCTS A TELEOPERATOR CLASS                   #############
#############################################################################

# Library Imports
import pygame
import numpy as np
from threading import Thread
import memory


#############################################################################
# KEYBOARD TELEOPERATION CLASS                                  #############
#############################################################################
class NumStick:
    def __init__(self, LinearPlanner, step_size=0.01):

        # Init. Pygame
        pygame.init()
        self.display = pygame.display.set_mode((400, 400))
        pygame.display.set_caption('Teleoperation: Keybaord')
        self.RenderText('Click Here! to Start')

        # Step Size
        self.step_size = step_size

        # Linear Planner
        self.LinearPlanner = LinearPlanner

    def run(self):
        self.controller = Thread(target=self.keyboard_listener)
        self.controller.start()

    def RenderText(self, msg):
        """
        Description:
            1. Renders a string on the pygame window.

        Args:
            msg -> string: Message to display on the window.
        """
        # Clear the Window
        self.display.fill((0, 0, 0))
        pygame.display.update()

        # Prepare the text
        green = (0, 255, 0)
        font = pygame.font.Font('freesansbold.ttf', 18)
        text = font.render(msg, True, green)
        textRect = text.get_rect()

        # Compute the center for the text
        textRect.center = (400 // 2, 400 // 2)

        # Refresh the Display
        self.display.blit(text, textRect)
        pygame.display.update()

    def keyboard_listener(self):
        """
        Description:
            1. Uses the keyboard for teleoperating the robot.
            2. Moves the robot with the fixed step size of 0.01m.
            3. KeyBindings,
                a. NumPad '6': +Y
                b. NumPad '4': -Y
                c. NumPad '8': -X
                d. NumPad '2': +X
                e. NumPad '7': -Y & -X
                f. NumPad '9': +Y & -X
                g. NumPad '1': +X & -Y
                h. NumPad '3': +X & +Y
                i. ArrowKey UP: +Z
                j. ArrowKey Down : -Z
        """

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.RenderText('Waiting for an Input!')

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP6:
                        print('Moving Robot: Direction +Y')
                        self.RenderText('Moving Robot: Direction +Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([0,
                                                         self.step_size,
                                                         0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP4:
                        print('Moving Robot: Direction -Y')
                        self.RenderText('Moving Robot: Direction -Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([0,
                                                         -self.step_size,
                                                         0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP8:
                        print('Moving Robot: Direction -X')
                        self.RenderText('Moving Robot: Direction -X')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([-self.step_size,
                                                         0,
                                                         0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP2:
                        print('Moving Robot: Direction +X')
                        self.RenderText('Moving Robot: Direction +X')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([self.step_size, 0, 0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_UP:
                        print('Moving Robot: Direction +Z')
                        self.RenderText('Moving Robot: Direction +Z')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([0, 0, self.step_size])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_DOWN:
                        print('Moving Robot: Direction -Z')
                        self.RenderText('Moving Robot: Direction -Z')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([0,
                                                         0,
                                                         -self.step_size])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP7:
                        print('Moving Robot: Direction -X & -Y')
                        self.RenderText('Moving Robot: Direction -X & -Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([-self.step_size,
                                                        -self.step_size, 0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP9:
                        print('Moving Robot: Direction -X & +Y')
                        self.RenderText('Moving Robot: Direction -X & +Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([-self.step_size,
                                                        self.step_size, 0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP1:
                        print('Moving Robot: Direction +X & -Y')
                        self.RenderText('Moving Robot: Direction +X & -Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([self.step_size,
                                                        -self.step_size, 0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_KP3:
                        print('Moving Robot: Direction +X & +Y')
                        self.RenderText('Moving Robot: Direction +X & +Y')

                        self.RenderText('Moving the Robot')
                        target = memory.pose + np.array([self.step_size,
                                                        self.step_size, 0])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')

                    if event.key == pygame.K_HOME:
                        print('Moving Robot: HOME')
                        self.RenderText('Moving Robot: HOME')

                        self.RenderText('Moving the Robot')
                        target = np.array([0.08, -0.05, -0.015])
                        self.LinearPlanner(target)
                        self.RenderText('Done!')
                        self.RenderText('Waiting for a New Input!')
