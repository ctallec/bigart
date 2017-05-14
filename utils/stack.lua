local classic = require 'classic'
local Stack = classic.class('Stack')

function Stack:_init(opts)
    self.memoryAllocation = opts.memoryAllocation

    local storageSize = torch.LongStorage(opts.cellSize:size(1)+1)
    storageSize[1] = self.memoryAllocation
    for i=1, opts.cellSize:size(1) do
        storageSize[i+1] = opts.cellSize[i]
    end
    self.storage = torch.Tensor(storageSize):zero()
    self.pointer = 0
end

function Stack:push(x)
    assert(self.pointer < self.memoryAllocation, "Stack full")
    self.pointer = self.pointer + 1
    self.storage[self.pointer]:copy(x)
end

function Stack:pop(buffer)
    assert(self.pointer > 0, "Stack empty")
    if buffer then
        buffer:copy(self.storage[self.pointer])
        self.pointer = self.pointer - 1
        return buffer
    else
        local buffer = self.storage[self.pointer]:clone()
        self.pointer = self.pointer - 1
        return buffer
    end
end

function Stack:peek(buffer)
    assert(self.pointer > 0, "Stack empty")
    if buffer then
        buffer:copy(self.storage[self.pointer])
        return buffer
    else
        local buffer = self.storage[self.pointer]:clone()
        return buffer
    end
end

function Stack:cuda()
    self.storage = self.storage:cuda()
    return self
end
-- local a1 = torch.Tensor(12,1):fill(3)
-- local a2 = torch.Tensor(12,1):fill(2)
-- local s = Stack({memoryAllocation=2, cellSize=a1:size()})
-- s:push(a1)
-- s:push(a2)
-- print(s:pop())
-- print(s:pop())

return Stack
