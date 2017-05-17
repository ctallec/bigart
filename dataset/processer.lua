local classic = require 'classic'
local Processer = classic.class('Processer')

function Processer:process(textfile, out_tensorfile, out_vocabfile)    
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('loading text file...')
    local f = torch.DiskFile(textfile)
    local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
    f:close()

    -- create vocabulary if it doesn't exist yet
    print('timer: ', timer:time().real)
    print('creating vocabulary mapping...')
    -- record all of them into a set
    local unordered = {}
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end

    -- sort them
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered) -- now order maps int->char

    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end

    -- construct a tensor with all the data
    print('timer: ', timer:time().real)
    print('putting data into tensor...')
    local data = torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
    end

    print('saving two files...')
    torch.save(out_vocabfile, vocab_mapping)
    torch.save(out_tensorfile, data)

    print('Done in time (seconds): ', timer:time().real)
end

function Processer:process_with_vocab(textfile, vocabfile, out_tensorfile)
    local vocab_mapping = torch.load(vocabfile)

    local f = torch.DiskFile(textfile)
    local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
    f:close()

    local data = torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
    end

    torch.save(out_tensorfile, data)
end

function Processer:processAndBatch(in_tensorfile, out_tensorfile, batch_size)
    local data = torch.load(in_tensorfile)
    local size = data:size(1)

    size = math.floor(size / batch_size) * batch_size
    data = data[{{1, size}}]:view(batch_size, size / batch_size):t()

    torch.save(out_tensorfile, data)
end

function Processer:split(tensorfile, test_nb_characters, batch_size, 
    train_tensorfile, valid_tensorfile, test_tensorfile)
    local data = torch.load(tensorfile)

    local train_data = data[{{1, -2*test_nb_characters-1}}]:clone()
    local valid_data = data[{{-2*test_nb_characters, -test_nb_characters-1}}]:clone()
    local test_data = data[{{-test_nb_characters, -1}}]:clone()

    local train_size = train_data:size(1)
    local valid_size = valid_data:size(1)
    local test_size = test_data:size(1)

    train_size = math.floor(train_size / batch_size) * batch_size
    valid_size = math.floor(valid_size / batch_size) * batch_size
    test_size = math.floor(test_size / batch_size) * batch_size

    train_data = train_data[{{1, train_size}}]:view(batch_size, train_size / batch_size):t()
    valid_data = valid_data[{{1, valid_size}}]:view(batch_size, valid_size / batch_size):t()
    test_data = test_data[{{1, test_size}}]:view(batch_size, test_size / batch_size):t()

    torch.save(train_tensorfile, train_data)
    torch.save(valid_tensorfile, valid_data)
    torch.save(test_tensorfile, test_data)
end

return Processer
