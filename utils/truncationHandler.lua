local classic = require 'classic'
local TruncationHandler = classic.class('TruncationHandler')

function TruncationHandler:_init(opts)
    self.t0 = opts.t0
end

function TruncationHandler:drawTruncation()
    return self.t0
end

function TruncationHandler:getReweighting(s)
    return 1
end

-- local trunc = TruncationHandler({t0=50, alpha=3})
-- for i=1, 10 do
--     print(trunc:drawTruncation())
-- end
-- 
-- print(trunc.truncationTable)
-- 
-- print(trunc:getReweighting(trunc.truncationTable[#trunc.truncationTable]))

return TruncationHandler
