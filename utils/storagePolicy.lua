local classic = require 'classic'
local StoragePolicy = classic.class("StoragePolicy")

function StoragePolicy:_init(opts)
    self.memoryAllocation = opts.memoryAllocation
    self.C = {}
    self.D = {}
end

function StoragePolicy:addTimestep()
    local t = #self.C + 1
    self.C[t] = {}
    self.D[t] = {}
    self.D[t][1] = math.max(t-1, 1)
    self.C[t][1] = t * (t+1) / 2
    for m = t, self.memoryAllocation do
        self.C[t][m] = 2 * t - 1
        self.D[t][m] = 1
    end

    for m=2, self.memoryAllocation do
        if t > m then
            local cmin
            for y = 1, t-1 do
                local c = y + self.C[y][m] + self.C[t-y][m-1]
                if not cmin or c < cmin then
                    cmin = c
                    self.D[t][m] = y
                end
            end
            self.C[t][m] = cmin
        end
    end
end

function StoragePolicy:addTimesteps(T)
    for t=1, T do
        self:addTimestep()
    end
end

function StoragePolicy:getCurrent()
    return #self.C
end

function StoragePolicy:setTimestep(T)
    local t = self:getCurrent()
    if T-t > 0 then
        self:addTimesteps(T-t)
    end
end

function StoragePolicy:tostring()
    local result = ""
    result = result .. "C:\n{"
    for t=1, #self.C do
        result = result .. "\n\t"
        for m=1, #self.C[t]-1 do
            result = result .. self.C[t][m] .. ", "
        end
        result = result .. self.C[t][#self.C[t]] .. ";"
    end
    result = result .. "\n}\n\n"
    result = result .. "D:\n{"
    for t=1, #self.D do
        result = result .. "\n\t"
        for m=1, #self.D[t]-1 do
            result = result .. self.D[t][m] .. ", "
        end
        result = result .. self.D[t][#self.D[t]] .. ";"
    end
    result = result .. "\n}"
    return result
end

-- local store = StoragePolicy{memoryAllocation=4}
-- store:setTimestep(10)
-- print(store:getCurrent())
-- print(store:tostring())

return StoragePolicy
