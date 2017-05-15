local classic = require 'classic'
local TruncationHandle = require 'utils.truncationHandle'
local ArtTruncationHandle = classic.class('ArtTruncationHandle', TruncationHandle)

function ArtTruncationHandle:_init(opts)
    self.truncationTable = {1}
    self.t0 = opts.t0
    self.alpha = opts.alpha
end

function ArtTruncationHandle:drawTruncation()
    local X_T=0
    local T=0
    repeat
        T=T+1
        c_T=self.alpha/((self.alpha-1)*self.t0 + T)
        X_T=torch.bernoulli(c_T)
    until(X_T==1)

    local cur_length = #self.truncationTable
    self.truncationTable[cur_length + 1] = self.truncationTable[cur_length] + T
    return T
end

function ArtTruncationHandle:getReweighting(s)
    local k = #self.truncationTable
    repeat
        T=self.truncationTable[k]
    until(T<=s)
    local c = self.alpha / ((self.alpha-1)*self.t0 + (s-T))
    io.read()
    return self.alpha / ((self.alpha-1)*self.t0 + (s-T)) 
end

-- local trunc = ArtTruncationHandle({t0=50, alpha=3})
-- for i=1, 10 do
--     print(trunc:drawTruncation())
-- end
-- 
-- print(trunc.truncationTable)
-- 
-- print(trunc:getReweighting(trunc.truncationTable[#trunc.truncationTable]))

return ArtTruncationHandle
