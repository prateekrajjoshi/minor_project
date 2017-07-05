# Vehicle Guidance using Image Processing

This project uses image processing to solve a puzzle (maze). A camera is mounted
over the puzzle which uses various image processing tools to solve the puzzle form ’start’ to
‘exit’. The program uses algorithm to find a shorter path to the ‘exit’. Then a vehicle (bot) is
guided through the maze to the ‘exit’.


# File Information

1) AVR.c contains the code related to bluetooth communication between the raspberry pi and the PC. 
2) route.py contains all the necessary opencv algorithms implemented in order to find the shortest path. Here, after determining the shortest path, the moving bot (vehicle) moves through the path and its live position is being recorded by the camera mounted depending upon which it sends information to the AVR connected to the vehicle regarding turning left, right, moving forward, stop and so on.
