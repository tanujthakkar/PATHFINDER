# PATHFINDER

The following project is mainly to study and compare different well-known and highly used path-planning algorithms. Path planning of a mobile robot is one of the basic operations needed to implement the navigation of the robot. It is one of the most researched topics in autonomous robotics. It is very important to find a safe path in a cluttered environment for a mobile robot. There are different approaches on which different path planning algorithms are based to handle varying situations, each of which have their own advantages and disadvantages for a certain situation. With this thought, path planning algorithms are analysed in this project in a custom 2D grid map, multiple runs are conducted to compare their performances of path optimality and execution time.

The following path-planning algorithms are studied and compared on the basis of path optimality and execution time. The implementations of these algorithms are <b>used and modified</b> from <a href="https://github.com/AtsushiSakai/PythonRobotics#path-planning">AtsushiSakai/PythonRobotics</a>.
<ul>
<li>Dijkstra</li>
<li>A*</li>
<li>Probabilistic Road Map</li>
<li>RRT</li>
</ul>

The algorithms were each implemented for four different goal cases with a starting position of (5,5) and one run was executed for a no solution case. The tests for the probabilistic algorithms, i.e., the probabilistic roadmap and RRT algorithms were run 10 times each for a single test and the average of the results were taken into consideration.
