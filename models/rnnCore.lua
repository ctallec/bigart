local classic = require 'classic'
local nn = require 'nn'
local RnnCore = classic.class('RnnCore')

function RnnCore:_init(opts)
    self.hiddenSize = opts.hiddenSize
    self.vocabSize = opts.vocabSize
end

function RnnCore:buildCore()
    local network = nn.Sequential()
    
    local input = nn.ParallelTable()
    input:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
    input:add(nn.Linear(self.hiddenSize, self.hiddenSize))

    network:add(input)
    network:add(nn.CAddTable())
    network:add(nn.Tanh())

    local output = nn.ConcatTable()

    local logprob = nn.Sequential()
    logprob:add(nn.Linear(self.hiddenSize, self.vocabSize))
    logprob:add(nn.LogSoftMax())

    output:add(logprob)
    output:add(nn.Identity())

    network:add(output)

    return network
end

-- local builder = RnnCore{hiddenSize=10, vocabSize=3}
-- local rnn = builder:buildCore()
-- print(unpack(rnn:forward{torch.Tensor{1, 2}, torch.Tensor(2, 10):zero()}))

return RnnCore
