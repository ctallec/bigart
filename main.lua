local paths = require 'paths'
local optim = require 'optim'


local Executer = require 'utils.executer'
local RnnCore = require 'models.rnnCore'
local Stack = require 'utils.stack'
local StoragePolicy = require 'utils.storagePolicy'

local cmd = torch.CmdLine()
cmd:option('-memoryAllocation', 100, 'memory allocation')
cmd:option('-truncation', 50, 'truncation')
cmd:option('-epochs', 20, 'number of epochs')
cmd:option('-cuda', false, 'gpu')
local opt = cmd:parse(arg)

torch.manualSeed(1)

local logdir = 'logs'
local modeldir = 'save'
local logfile = cmd:string(paths.concat(logdir, 'log'), opt, {cuda=true, epochs=true,
    memoryAllocation=true})
local modelfile = cmd:string(paths.concat(modeldir, 'model'), opt, {cuda=true, 
    epochs=true, memoryAllocation=true}) .. '.t7'
local logstream = io.open(logfile, 'w')

local dataset_directory = 'dataset'
local train_file = paths.concat(dataset_directory, 'train.t7')
local valid_file = paths.concat(dataset_directory, 'valid.t7')
local test_file = paths.concat(dataset_directory, 'test.t7')
local vocab_file = paths.concat(dataset_directory, 'vocab_text8.t7')

local train_data = torch.load(train_file)
local valid_data = torch.load(valid_file)
local vocab = torch.load(vocab_file)

local vocab_size = 0
for k,v in pairs(vocab) do
    vocab_size = vocab_size + 1
end

local train_size = train_data:size(1)
local valid_size = valid_data:size(1)

local hiddenSize = 512
local batchSize = train_data:size(2)

local rnnBuilder = RnnCore{vocabSize=vocab_size, hiddenSize=hiddenSize, rnnType='lstm'}
local rnn = rnnBuilder:buildCore()

local stateSize = rnnBuilder:getStateSize()

local initState = torch.Tensor(batchSize, stateSize):zero()
local hiddenGradient = torch.Tensor(batchSize, stateSize):zero()
local nb_sequences = math.floor(train_size / opt.truncation)

local stack = Stack{memoryAllocation=opt.memoryAllocation, 
    cellSize=torch.Tensor{batchSize, stateSize}}
stack:push(initState)

local policy = StoragePolicy{memoryAllocation=opt.memoryAllocation}
policy:addTimesteps(opt.truncation)

local exec = Executer{D=policy.D,stack=stack,rnnCore=rnn}

local criterion = nn.ClassNLLCriterion()
local loss = 0

local optimState = {
    learningRate=1e-3
}

if opt.cuda then
    require 'cunn'
    require 'cudnn'

    initState = initState:cuda()
    hiddenGradient = hiddenGradient:cuda()
    train_data = train_data:cuda()
    valid_data = valid_data:cuda()
    stack = stack:cuda()
    rnn = rnn:cuda()
    criterion = criterion:cuda()
end

local params, gradParams = rnn:getParameters()

function exec:getInput(t)
    return train_data[(t-1)%train_size + 1]
end

function exec:setOutputAndGetGradOutput(t, output)
    loss = loss + criterion:forward(output, train_data[t%train_size + 1])
    return criterion:backward(output, train_data[t%train_size + 1])
end

local function train()
    local t=1
    local cumLoss = 0

    local function feval(params_)
        if params_ ~= params then
            params:copy(params_)
        end

        loss = 0
        gradParams:zero()
        exec:executeStrategy(hiddenGradient, opt.memoryAllocation-1,
            opt.truncation, t)

        return loss, gradParams
    end

    for i=1, nb_sequences do
        xlua.progress(i, nb_sequences)
        local _, batch_loss = optim.rmsprop(feval, params, optimState)
        cumLoss = cumLoss + batch_loss[1]
        t = t + opt.truncation
    end
    return cumLoss / nb_sequences / opt.truncation
end

local function evaluate()
    local t=1
    local cumLoss = 0
    local state = torch.Tensor(batchSize, stateSize):zero()

    if opt.cuda then
        state = state:cuda()
    end

    for t=1, valid_size do
        xlua.progress(t, valid_size)
        local output, next_state = unpack(rnn:forward{valid_data[t], state})
        state:copy(next_state)
        cumLoss = cumLoss + criterion:forward(output, valid_data[t%valid_size+1])
    end

    return cumLoss / valid_size
end

for e=1, opt.epochs do
    torch.save(modelfile, rnn)
    print("On epoch " .. e .. ":")
    print("Train:")
    local train_loss = train()
    print(train_loss)
    print("Validation:")
    local validation_loss = evaluate()
    print(validation_loss)
    logstream:write(e .. ' ' .. train_loss .. ' ' .. validation_loss .. '\n')
end
