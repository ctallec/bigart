local Processer = require 'dataset.processer'

local cmd = torch.CmdLine()
cmd:option('-batch_size', 32, 'batch size')

local opt = cmd:parse(arg)

local directory = 'dataset/ptb/'
local train_file = directory .. 'train.txt'
local test_file = directory .. 'test.txt'
local valid_file = directory .. 'valid.txt'
local tensor_train_file = directory .. 'train.t7'
local tensor_test_file = directory .. 'test.t7'
local tensor_valid_file = directory .. 'valid.t7'
local vocab_file = directory .. 'vocab.t7'

local proc = Processer()
proc:process(valid_file, tensor_valid_file, vocab_file)
proc:process(test_file, tensor_test_file, vocab_file)
proc:process(train_file, tensor_train_file, vocab_file)

proc:processAndBatch(tensor_valid_file, tensor_valid_file, opt.batch_size)
proc:processAndBatch(tensor_train_file, tensor_train_file, opt.batch_size)
proc:processAndBatch(tensor_test_file, tensor_test_file, opt.batch_size)
