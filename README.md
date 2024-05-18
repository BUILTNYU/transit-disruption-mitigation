# transit-disruption-mitigation
## Description
An algorithm for transit disruption mitigation - strategy seletion phase (resource allocation)[^1]

[^1]: some of the code is designed specificaly for example 1; changes are needed if you want to apply it to other networks!

## Usage
Run <i>main.py [Q_scenario] [q_0] [q_max] [T_scenario] [alpha]</i> <br>
@args Q_scenario: the name of demand pattern <br>
@args q_0, q_max: specification the level of demand, like "uniform", "convex", "concave", "increasing", "decreasing" <br>
@args T_scenario: the name of the disruption distribution, like "uniform", "exponential", "normal", "Dirac_0", "Dirac_Tub", "bi_Dirac" <br>

demand pattern illustration: <br>
 <img src="img/demand_patterns.png" width = "400" alt="evaluation_prog" align=center />

disruption distribution illustration: <br>
 <img src="img/disruption_distributions.png" width = "400" alt="evaluation_prog" align=center />

## Components
The evaluation module takes time-dependent demand, disruption distribution, and mitigation plans as input, and outputs the user and operator cost in the horizon. Time is discretized into one minute intervals when accumulating the user costs. The user demands in each one-minute interval are assigned according to the capacity constraints at that time. Not enough capacity on the shortest
path means that users will detour to longer distance paths. User wait cost depends on average headway.

 <img src="img/evaluation_program.png" width = "600" alt="evaluation_prog" align=center />

User cost computation:

 <img src="img/evaluation_formula.png" width = "200" alt="user_cost_comp" align=center />

## Contributors
Qi Liu(ql375@nyu.edu), Joseph Chow(joseph.chow@nyu.edu)
