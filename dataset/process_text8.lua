local Processer = require 'dataset.processer'

local cmd = torch.CmdLine()
cmd:option('-batch_size', 128, 'batch size')

local opt = cmd:parse(arg)

local directory = 'dataset/'
local text_file = directory .. 'text8'
local tensor_text_file = directory .. 'text8.t7'
local vocab_file = directory .. 'vocab_text8.t7'
local train_file = directory .. 'train.t7'
local valid_file = directory .. 'valid.t7'
local test_file = directory .. 'test.t7'
local test_nb_characters = 5e6

local proc = Processer()
proc:process(text_file, tensor_text_file, vocab_file)
proc:split(tensor_text_file, test_nb_characters, opt.batch_size,
    train_file, valid_file, test_file)
