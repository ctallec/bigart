local nngraph = require 'nngraph'
local paths = require 'paths'

local cmd = torch.CmdLine()
cmd:option('-rnnType', 'lstm', 'network type')
cmd:option('-hiddenSize', 32, 'hidden size')
cmd:option('-samples', 500, 'number of samples')
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

local rnn = torch.load('save/model.t7')

local state = torch.Tensor(1, stateSize):zero()
local input = torch.Tensor{1}

for i=1, opt.samples do
    io.write(inv_vocab[input[1]])
    local out, next_state = unpack(rnn:forward{input, state})
    local next_index = torch.multinomial(out:exp():div(out:sum()), 1):squeeze()
    input[1] = next_index
    state:copy(next_state)
end
