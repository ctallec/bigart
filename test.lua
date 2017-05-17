local paths = require 'paths'
local nngraph = require 'nngraph'

local cmd = torch.CmdLine()
cmd:option('-model', 'save/model.t7', 'model file')
cmd:option('-cuda', false, 'gpu')
cmd:option('-hiddenSize', 512, 'number of hidden units')
local opt = cmd:parse(arg)

local sizeTable = {
    rnn=opt.hiddenSize,
    lstm=opt.hiddenSize*2,
    mlstm=opt.hiddenSize*2
}

local statesize = sizeTable['lstm']

local dataset_directory = 'dataset/ptb'
local test_file = paths.concat(dataset_directory, 'test.t7')

local test_data = torch.load(test_file)
local test_size = test_data:size(1)
local batchsize = test_data:size(2)

local criterion = nn.ClassNLLCriterion()

if opt.cuda then
    require 'cudnn'
    require 'cunn'
    require 'cutorch'

    test_data = test_data:cuda()
    criterion = criterion:cuda()
end

local rnn = torch.load(opt.model)

local function evaluate()
    local t=1
    local cumloss = 0
    local state = torch.Tensor(batchsize, statesize):zero()

    if opt.cuda then
        state = state:cuda()
    end

    for t=1, test_size do
        xlua.progress(t, test_size)
        local output, next_state = table.unpack(rnn:forward{test_data[t], state})
        state:copy(next_state)
        cumloss = cumloss + criterion:forward(output, test_data[t%test_size+1])
    end

    return cumloss / test_size / math.log(2)
end

print('Test loss: ' .. evaluate())
