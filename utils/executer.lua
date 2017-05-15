local classic = require 'classic'
local Executer = classic.class('Executer')

function Executer:_init(opts)
    self.D = opts.D
    self.rnnCore = opts.rnnCore
    self.stack = opts.stack

--d    self.count = 0
end

function Executer:executeStrategy(gradHidden, m, t, s)
    local hiddenState = self.stack:peek()
    if t == 0 then
        return gradHidden
    elseif t == 1 then
        local output, _ = unpack(self.rnnCore:forward({self:getInput(s), hiddenState}))
--d        self.count = self.count + 1
        local gradOutput = self:setOutputAndGetGradOutput(s+t-1, output)
        gradHidden:mul(1/(1-self:getReweighting(s)))
        local gradInput, gradHiddenPrevious = unpack(self.rnnCore:backward(
            {self:getInput(s), hiddenState},
            {gradOutput, gradHidden}))
        return gradHiddenPrevious
    else
        local y = self.D[t][m]
        for i=0, y-1 do
            local output, tempHiddenState = unpack(self.rnnCore:forward({self:getInput(s+i), hiddenState}))
--d            self.count = self.count + 1
            hiddenState:copy(tempHiddenState)
        end
        self.stack:push(hiddenState)
        local gradHiddenR = self:executeStrategy(gradHidden, m-1, t-y, s+y)
        self.stack:pop()
        local gradHiddenL = self:executeStrategy(gradHiddenR, m, y, s)
        return gradHiddenL
    end
end

function Executer:getReweighting(s)
    return 1
end

return Executer
