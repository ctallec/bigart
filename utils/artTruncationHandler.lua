local classic = require 'classic'
local TruncationHandler = require 'utils.truncationHandler'
local ArtTruncationHandler = classic.class('ArtTruncationHandler', TruncationHandler)

function ArtTruncationHandler:_init(opts)
    self.truncationTable = {1}
    self.t0 = opts.t0
    self.alpha = opts.alpha
end

function ArtTruncationHandler:drawTruncation()
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

function ArtTruncationHandler:getReweighting(s)
    local k = #self.truncationTable
    repeat
        T=self.truncationTable[k]
        k=k-1
    until(T<=s)
    return self.alpha / ((self.alpha-1)*self.t0 + (s-T)) 
end

-- local trunc = ArtTruncationHandler({t0=50, alpha=3})
-- for i=1, 10 do
--     print(trunc:drawTruncation())
-- end
-- 
-- print(trunc.truncationTable)
-- 
-- print(trunc:getReweighting(trunc.truncationTable[#trunc.truncationTable]))

return ArtTruncationHandler
