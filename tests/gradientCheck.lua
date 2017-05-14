local optim = require 'optim'
local Executer = require 'utils.executer'
local RnnCore = require 'models.rnnCore'
local Stack = require 'utils.stack'
local StoragePolicy = require 'utils.storagePolicy'

local memoryAllocation = 5
local temporality = 10
local vocabSize = 5
local hiddenSize = 3

local rnnBuilder = RnnCore{vocabSize=vocabSize, hiddenSize=hiddenSize}
local rnn = rnnBuilder:buildCore()
local params, gradParams = rnn:getParameters()

local stack = Stack{memoryAllocation=memoryAllocation, cellSize=torch.Tensor{2, hiddenSize}}
stack:push(torch.Tensor(2, hiddenSize):fill(0))

local policy = StoragePolicy{memoryAllocation=memoryAllocation}

for i=1, temporality do
    policy:addTimestep()
end

local exec = Executer{D=policy.D,stack=stack,rnnCore=rnn}

local criterion = nn.ClassNLLCriterion()

local inputs = {
    torch.Tensor{1, 2},
    torch.Tensor{2, 3},
    torch.Tensor{3, 4},
    torch.Tensor{4, 5}, 
    torch.Tensor{5, 1},
    torch.Tensor{1, 2},
    torch.Tensor{2, 3},
    torch.Tensor{3, 4},
    torch.Tensor{4, 5}, 
    torch.Tensor{5, 1},
}

local targets = {
    torch.Tensor{2, 3},
    torch.Tensor{3, 4},
    torch.Tensor{4, 5}, 
    torch.Tensor{5, 1},
    torch.Tensor{1, 2},
    torch.Tensor{2, 3},
    torch.Tensor{3, 4},
    torch.Tensor{4, 5}, 
    torch.Tensor{5, 1},
    torch.Tensor{1, 2},
}

function exec:getInput(t)
    return inputs[t]
end

local loss = 0

function exec:setOutputAndGetGradOutput(t, output)
    loss = loss + criterion:forward(output, targets[t])
    return criterion:backward(output, targets[t])
end

local function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    
    loss = 0
    gradParams:zero()
    exec:executeStrategy(torch.Tensor(2, hiddenSize):zero(), memoryAllocation-1,
        temporality, 1)

    return loss, gradParams
end

feval(params)
print(exec.count)
print(policy.C[temporality][memoryAllocation-1])
print(optim.checkgrad(feval, params))
print(stack.storage)
