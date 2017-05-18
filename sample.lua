local nngraph = require 'nngraph'
local paths = require 'paths'

local cmd = torch.CmdLine()
cmd:option('-rnnType', 'lstm', 'network type')
cmd:option('-hiddenSize', 32, 'hidden size')
cmd:option('-samples', 500, 'number of samples')
cmd:option('-cuda', false, 'gpu')
cmd:option('-model', 'save/model.t7', 'model file')
local opt = cmd:parse(arg)

local dataset_directory = 'dataset/ptb'
local vocab_file = paths.concat(dataset_directory, 'vocab.t7')
local vocab = torch.load(vocab_file)

local inv_vocab = {}
for k, v in pairs(vocab) do
    inv_vocab[v] = k
end

local sizeTable = {
    lstm = 2 * opt.hiddenSize,
    rnn = opt.hiddenSize,
    mlstm = 2 * opt.hiddenSize
}

local stateSize = sizeTable[opt.rnnType]

local state = torch.Tensor(1, stateSize):zero()
local input = torch.Tensor{1}

if opt.cuda then
    require 'cudnn'
    require 'cunn'
    require 'cutorch'

    state = state:cuda()
    input = input:cuda()
end

local rnn = torch.load(opt.model)

for i=1, opt.samples do
    io.write(inv_vocab[input[1]])
    local out, next_state = table.unpack(rnn:forward{input, state})
    local next_index = torch.multinomial(out:exp():div(out:sum()), 1):squeeze()
    input[1] = next_index
    state:copy(next_state)
end
