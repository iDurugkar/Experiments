function [Q, iters] = Qlearn_fourier(num_episodes, num_features)

lambda = 0.9;
explore_rate = 0.0;
discount = 1;
learning_rate = 0.1;

POS_RANGE = [-1.20, 0.5];
VEL_RANGE = [-0.07, 0.07];

features = fourier_features(num_features);
actions = [1, 2, 3];

weights = zeros(num_features, length(actions));

iters = zeros(num_episodes, 1);
for i = 1:num_episodes
    figure(1);
    hold on;
    traces = zeros(num_features, length(actions));
    [state, ~, simEnd] = mcar_simulation();
    action = actions(randi(length(actions)));
    oldF = fourier_approx(state);
    iter = 0;
    while ~simEnd
        [newstate, reward, simEnd] = mcar_simulation(state, action);
        newF = fourier_approx(newstate);
        
        [maxQ, nextaction] = max(approxQ(newF, actions));
        
        delta = reward - approxQ(oldF, action);
        
        if ~simEnd, delta = delta + discount*maxQ;
        end
        
        traces = discount*lambda*traces;
        traces(:, action) = traces(:, action) + oldF;
        
        weights = weights + learning_rate*delta*traces;
        
        if rand()>explore_rate
            action = nextaction;
        else
            action = actions(randi(length(actions)));
        end
        state = newstate;
        iter = iter + 1;
        oldF = newF;
        if action == 1
            plot(state(1), state(2),'bo');
        elseif action == 2
            plot(state(1), state(2),'rx');
        else
            plot(state(1), state(2),'g+');
        end
    end
    iters(i) = iter;
    fprintf('Episode %d took %d iterations\n',i,iter);
end
Q = generateQ();
return;

%%% Functions %%%

function Q = generateQ()
    Q = zeros(300,20,3);
    [ps,vs] = meshgrid(linspace(-1.2,0.5,size(Q,1)),linspace(-0.07,0.07,size(Q,2)));
    for p = 1:size(Q,1)
        for v = 1:size(Q,2)
            Q(p,v,actions) = approxQ(project([ps(p),vs(v)]),actions);
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