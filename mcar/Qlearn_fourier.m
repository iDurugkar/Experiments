function [Q, iters] = Qlearn_fourier(num_episodes, num_features)

discount = 1;
lambda = 0.9;
learning_rate = 0.01;
exploration_rate = 0;

POS_RANGE = [-1.2, 0.5];
VEL_RANGE = [-0.07, 0.07];

features = fourier_features(num_features);

actions = [1, 2, 3];
weights = zeros(num_features, length(actions));
iters = zeros(num_episodes,1);
toPlot = 0;
for episode = 1:num_episodes
    iter = 0;
    if toPlot
        figure(episode);
        hold on;
    end
    [state, ~, isDone] = mcar_simulation();
    action = randsample(actions,1,true);
    traces = zeros(num_features, length(actions));
    
    oldF = fourier_approx(state);
    while ~isDone
        [newstate, reward, isDone] = mcar_simulation(state, action);
        newF = fourier_approx(newstate);
        
        [maxQ, nextAction] = max(approxQ(newF, actions));
        delta = reward - approxQ(oldF, action);
        
        if ~isDone, delta = delta + maxQ; end
        
        traces = discount*traces*lambda;
        traces(:,action) = traces(:,action) + oldF;
        
        weights = weights + learning_rate*delta*traces;
        
        if toPlot
            if action == 1
                plot(state(1), state(2), 'bo');
            elseif action == 2
                plot(state(1), state(2), 'rx');
            else
                plot(state(1), state(2), 'g+');
            end
        end
        
        if rand() > exploration_rate
            action = nextAction;
        else
            action = randsample(actions,1);
        end
        state = newstate;
        oldF = newF;
        iter = iter+1;
    end
    iters(episode) = iter;
    fprintf('Episode %d completed in %d iterrations\n',episode, iter);
end
Q = generateQ();
return
%%% Functions %%%

function Q = generateQ()
    Q = zeros(300,20,3);
    [ps,vs] = meshgrid(linspace(-1.2,0.5,size(Q,1)),linspace(-0.07,0.07,size(Q,2)));
    for p = 1:size(Q,1)
        for v = 1:size(Q,2)
            Q(p,v,actions) = approxQ(fourier_approx([ps(p),vs(v)]),actions);
        end
    end
end

function aQ = approxQ(F, actions)
    w = weights(:,actions);
    aQ = sum(w.*repmat(F,1,length(actions)));
end
function F = fourier_approx(s)
    srange = diff([POS_RANGE; VEL_RANGE]');
    ss = (s - [POS_RANGE(1), VEL_RANGE(1)])./srange;
    F = cos(pi * sum(repmat(ss,num_features,1).*features,2));
end
function features = fourier_features(n)
    order = floor(sqrt(n)-1);
    [x,y] = meshgrid(0:order,0:order);
    features = [y(:),x(:)];
    features = features(1:n,:);
end
end