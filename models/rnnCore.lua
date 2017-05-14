local nn = require 'nn'
local classic = require 'classic'
local nngraph = require 'nngraph'
local RnnCore = classic.class('RnnCore')

function RnnCore:_init(opts)
    self.hiddenSize = opts.hiddenSize
    self.vocabSize = opts.vocabSize
    self.rnnType = opts.rnnType or 'rnn'
end

function RnnCore:buildCore()
    local rnnTable = {
        rnn = self.buildRNN,
        lstm = self.buildLSTM,
        mlstm = self.buildMLSTM
    }
    return rnnTable[self.rnnType](self)
end

function RnnCore:getStateSize()
    local sizeTable = {
        rnn = self.hiddenSize,
        lstm = 2 * self.hiddenSize,
        mlstm = 2 * self.hiddenSize
    }
    return sizeTable[self.rnnType]
end

function RnnCore:buildRNN()
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

function RnnCore:buildLSTM()
    local x = nn.Identity()()
    local s = nn.Identity()()
    local h = nn.Narrow(-1, 1, self.hiddenSize)(s)
    local c = nn.Narrow(-1, self.hiddenSize+1, self.hiddenSize)(s)

    local m = nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(h)
    }

    local hhat = nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    }

    local i = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local o = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local f = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local next_c = nn.CAddTable(){
        nn.CMulTable(){f, c},
        nn.CMulTable(){i, hhat}
    }

    local next_h = nn.Tanh()(nn.CMulTable(){next_c, o})

    local next_s = nn.JoinTable(-1){next_h, next_c}

    local output = nn.LogSoftMax()(nn.Linear(self.hiddenSize, self.vocabSize)(next_h))

    local network = nn.gModule({x, s}, {output, next_s})
    return network
end

function RnnCore:buildMLSTM()
    local x = nn.Identity()()
    local s = nn.Identity()()
    local h = nn.SelectTable({{},{1, self.hiddenSize}})(s)
    local c = nn.SelectTable({{},{self.hiddenSize, -1}})(s)

    local m = nn.CMulTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(h)
    }

    local hhat = nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    }

    local i = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local o = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local f = nn.Sigmoid()(nn.CAddTable(){
        nn.LookupTable(self.vocabSize, self.hiddenSize)(x),
        nn.Linear(self.hiddenSize, self.hiddenSize)(m)
    })

    local next_c = nn.CAddTable(){
        nn.CMulTable(){f, c},
        nn.CMulTable(){i, hhat}
    }

    local next_h = nn.Tanh()(nn.CMulTable(){next_c, o})

    local next_s = nn.JoinTable(-1){next_h, next_c}

    local output = nn.LogSoftMax()(nn.Linear(self.hiddenSize, self.vocabSize)(next_h))

    local network = nn.gModule({x, s}, {o, next_s})
    return network
end

-- local builder = RnnCore{hiddenSize=10, vocabSize=3}
-- local rnn = builder:buildCore()
-- print(unpack(rnn:forward{torch.Tensor{1, 2}, torch.Tensor(2, 10):zero()}))

return RnnCore
