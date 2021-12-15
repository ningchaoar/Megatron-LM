from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, GPT2Config


"""
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
          Model Name             Batch Size     Seq Length     Time in s   
--------------------------------------------------------------------------------
             gpt2                    1               1             0.014     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
          Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
             gpt2                    1               1              N/A      
--------------------------------------------------------------------------------
"""

# latency for generating 1 token
args = PyTorchBenchmarkArguments(models=["gpt2"], batch_sizes=[1], sequence_lengths=[1])
config_base = GPT2Config()
benchmark = PyTorchBenchmark(args, configs=[config_base])
benchmark.run()
