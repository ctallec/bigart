local paths = require 'paths'
local optim = require 'optim'


local Executer = require 'utils.executer'
local RnnCore = require 'models.rnnCore'
local Stack = require 'utils.stack'
local StoragePolicy = require 'utils.storagePolicy'

local cmd = torch.CmdLine()
cmd:option('-memoryAllocation', 200, 'memory allocation')
cmd:option('-truncation', 50, 'truncation')
cmd:option('-epochs', 20, 'number of epochs')
cmd:option('-cuda', false, 'gpu')
cmd:option('-gpu', 1, 'index of gpu')
cmd:option('-tbptt', false, 'using art or truncated bptt')
cmd:option('-learningRate', 3e-4, 'learning rate')
cmd:option('-hiddenSize', 512, 'number of hidden units')
local opt = cmd:parse(arg)

local TruncationHandlerFile = opt.tbptt and 'utils.truncationHandler' or 'utils.artTruncationHandler'
local TruncationHandler = require(TruncationHandlerFile)
local trunc = TruncationHandler({t0=opt.truncation, alpha=3})

torch.manualSeed(1)

local logdir = 'logs'
local modeldir = 'save'
local logfile = cmd:string(paths.concat(logdir, 'log'), opt, {cuda=true, epochs=true,
    memoryAllocation=true})
local modelfile = cmd:string(paths.concat(modeldir, 'model'), opt, {cuda=true, 
    epochs=true, memoryAllocation=true}) .. '.t7'
local logstream = io.open(logfile, 'w')

local dataset_directory = 'dataset/ptb'
local train_file = paths.concat(dataset_directory, 'train.t7')
local valid_file = paths.concat(dataset_directory, 'valid.t7')
local test_file = paths.concat(dataset_directory, 'test.t7')
local vocab_file = paths.concat(dataset_directory, 'vocab.t7')

local train_data = torch.load(train_file)
local valid_data = torch.load(valid_file)
local vocab = torch.load(vocab_file)

local vocab_size = 0
for k,v in pairs(vocab) do
    vocab_size = vocab_size + 1
end

local train_size = train_data:size(1)
local valid_size = valid_data:size(1)

local hiddenSize = opt.hiddenSize
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
function exec:getReweighting(s)
    return trunc:getReweighting(s)
end

local criterion = nn.ClassNLLCriterion()
local loss = 0

local optimState = {
    learningRate=3e-4
}

if opt.cuda then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'

    cutorch.setDevice(opt.gpu)
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
    local T=0

    local function feval(params_)
        if params_ ~= params then
            params:copy(params_)
        end

        loss = 0
        gradParams:zero()
        T = trunc:drawTruncation()
        policy:setTimestep(T)
        exec:executeStrategy(hiddenGradient, opt.memoryAllocation-1,
            T, t)

        return loss, gradParams
    end

    repeat
        io.write('\rBatch: ' .. t .. '/' .. train_size .. ' -- Current loss: ' .. cumLoss / t / math.log(2))
        io.flush()
        local _, batch_loss = optim.rmsprop(feval, params, optimState)
        cumLoss = cumLoss + batch_loss[1]
        t = t + T
    until(t>=train_size)
    return cumLoss / t / math.log(2)
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
        local output, next_state = table.unpack(rnn:forward{valid_data[t], state})
        state:copy(next_state)
        cumLoss = cumLoss + criterion:forward(output, valid_data[t%valid_size+1])
    end

    return cumLoss / valid_size / math.log(2)
end

local min_loss = 10000
local validation_loss = 1000
for e=1, opt.epochs do
    if validation_loss < min_loss then
        torch.save(modelfile, rnn)
        min_loss = validation_loss
    end
    print("On epoch " .. e .. ":")
    print("Train:")
    local train_loss = train()
    trunc:reset()
    print(train_loss)
    print("Validation:")
    validation_loss = evaluate()
    print(validation_loss)
    logstream:write(e .. ' ' .. train_loss .. ' ' .. validation_loss .. '\n')
    logstream:flush()
end
