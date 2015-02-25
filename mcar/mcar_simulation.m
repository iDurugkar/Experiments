function [nextstate, reward, simend] = mcar_simulation(state, action)
%Simulator for mountain car

prange = [-1.2, 0.5];
vrange = [-0.07 0.07];

goal = 0.5;
gravity = -0.0025;
dT = 0.001;

if nargin < 1
    nextstate = [-0.5, 0];
    reward = 0;
    simend = 0;
    return
elseif nargin < 2
    nextstate = state;
    reward = 0;
    simend = 0;
    return
end

action_values = [0, -1, 1];

%new velocity
newv = state(2) + dT*action_values(action) + gravity*cos(3*state(1));
%bounded
newv = min(vrange(2), max(vrange(1),newv));

newp = state(1) + newv;
if newp < prange(1)
    newp = prange(1);
    newv = 0;
elseif newp > prange(2)
    newp = prange(2);
    newv = 0;
end

nextstate = [newp, newv];

if nextstate(1) >= goal
    simend = 1;
    reward = 0;
else
    simend = 0;
    reward = -1;
end

return
